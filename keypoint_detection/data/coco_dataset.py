import argparse
import json
import math
import typing
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
import cv2

import albumentations as A
import torch
torch.set_float32_matmul_precision("medium")
import numpy as np
from torchvision.transforms import ToTensor

from keypoint_detection.data.coco_parser import CocoImage, CocoKeypointCategory, CocoKeypoints
from keypoint_detection.data.imageloader import ImageDataset, ImageLoader
from keypoint_detection.types import COCO_KEYPOINT_TYPE, IMG_KEYPOINTS_TYPE


class COCOKeypointsDataset(ImageDataset):
    """Pytorch Dataset for COCO-formatted Keypoint dataset

    cf. https://cocodataset.org/#format-data for more information. We expect each annotation to have at least the keypoints and num_keypoints fields.
    Each category should also have keypoints. For more information on the required fields and data types, have a look at the COCO parser in `coco_parser.py`.

    The dataset builds an index during the init call that maps from each image_id to a list of all keypoints of all semantic types in the dataset.

    The Dataset also expects a keypoint_channel_configuration that maps from the semantic types (the keypoints in all categories of the COCO file) to the channels
    of the keypoint detector. In the simplest case this is simply a list of all types, but for e.g. symmetric objects or equivalence mapping one could combine different
    types into one channel. For example if you have category box with keypoints [corner0, corner1, corner2, corner3] you could combine  them in a single channel for the
    detector by passing as configuration [[corner0,corner1,corner2,corner3]].

    You can also select if you want to train on annotations with flag=1 (occluded).

    The paths in the JSON should be relative to the directory in which the JSON is located.


    The __getitem__ function returns [img_path, [keypoints for each channel according to the configuration]]
    """

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        """
        parser = parent_parser.add_argument_group("COCOkeypointsDataset")
        parser.add_argument(
            "--detect_only_visible_keypoints",
            dest="detect_only_visible_keypoints",
            default=False,
            action="store_true",
            help="If set, only keypoints with flag > 1.0 will be used.",
        )
        
        parser.add_argument(
            "--max_image_size",
            type=int,
            dest="max_image_size",
            default=512,
            help="Maximum image size for the images in the dataset. Images will be resized to this size if they are larger.",
        )

        return parent_parser

    def __init__(
        self,
        json_dataset_path: str,
        keypoint_channel_configuration: list[list[str]],
        detect_only_visible_keypoints: bool = True,
        transform: A.Compose = None,
        imageloader: ImageLoader = None,
        max_image_size: int = 512,
        **kwargs,
    ):
        super().__init__(imageloader)

        self.image_to_tensor_transform = ToTensor()
        self.dataset_json_path = Path(json_dataset_path)
        self.dataset_dir_path = self.dataset_json_path.parent  # assume paths in JSON are relative to this directory!

        self.keypoint_channel_configuration = keypoint_channel_configuration
        self.detect_only_visible_keypoints = detect_only_visible_keypoints
        
        # Adjust max_image_size to be divisible by 32
        divisor = 32
        original_max_size = max_image_size
        max_image_size = (max_image_size // divisor) * divisor
        if max_image_size != original_max_size:
            print(f"Warning: Adjusted max_image_size from {original_max_size} to {max_image_size} to ensure divisibility by {divisor}")
        self.max_image_size = max_image_size


        self.random_crop_transform = None
        self.transform = transform
        self.dataset = self.prepare_dataset()  # idx: (image, list(keypoints/channel))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, IMG_KEYPOINTS_TYPE]:
        """
        Returns:
            (image, keypoints); image = 3xHxW tensor; keypoints = List(c x list( list of K_i keypoints ))

            e.g. for 2 heatmap channels with respectively 1,2 keypoints, the keypoints list will be formatted as
            [[[u11,v11]],[[u21,v21],[u22,v22]]]
        """
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image_path = self.dataset_dir_path / self.dataset[index][0]
        image = self.image_loader.get_image(str(image_path), index)
        # turn grayscale image to 3-channel
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
         # remove a-channel if needed
        if image.shape[2] == 4:
            image = image[..., :3]
        longest_side = max(image.shape[0], image.shape[1])
        
        scale = self.max_image_size / longest_side

        
        # Ensure valid dimensions
        new_height = min(int(image.shape[0] * scale), self.max_image_size)
        new_width = min(int(image.shape[1] * scale), self.max_image_size)
        
        # Calculate scaling ratios for keypoints
        x_ratio = new_width / image.shape[1]
        y_ratio = new_height / image.shape[0]
        
        # Resize image
        #image = A.resize(image, (new_height, new_width), interpolation=1)
        #resize using opencv
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
        #we need to add padding to have a square image, will be used for collate_fn, original image will be placed at top left so we don't have to deal with padding in the heatmap generation
        black_background = np.zeros((self.max_image_size, self.max_image_size, 3), dtype=np.uint8)
        black_background[:new_height, :new_width] = image
        image = black_background

        keypoints = self.dataset[index][1]
        
        # Scale keypoints to match the resized image
        keypoints_np = []
        for channel_keypoints in keypoints:
            scaled_channel_keypoints = []
            for kp in channel_keypoints:
                scaled_kp = [kp[0] * x_ratio, kp[1] * y_ratio]
                scaled_channel_keypoints.append(scaled_kp)
            keypoints_np.append(scaled_channel_keypoints)
        keypoints = keypoints_np
        

        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image, keypoints = transformed["image"], transformed["keypoints"]

        # convert all keypoints to integers values.
        # COCO keypoints can be floats if they specify the exact location of the keypoint (e.g. from CVAT)
        # even though COCO format specifies zero-indexed integers (i.e. every keypoint in the [0,1]x [0.1] pixel box becomes (0,0)
        # we convert them to ints here, as the heatmap generation will add a 0.5 offset to the keypoint location to center it in the pixel
        # the distance metrics also operate on integer values.

        # so basically from here on every keypoint is an int that represents the pixel-box in which the keypoint is located.
        keypoints = [
            [[math.floor(keypoint[0]), math.floor(keypoint[1])] for keypoint in channel_keypoints]
            for channel_keypoints in keypoints
        ]
        image = self.image_to_tensor_transform(image)
        return image, keypoints

    def prepare_dataset(self):  # noqa: C901
        """Prepares the dataset to map from COCO to (img, [keypoints for each channel])

        Returns:
            [img_path, [list of keypoints for each channel]]
        """
        with open(self.dataset_json_path, "r") as file:
            data = json.load(file)
            parsed_coco = CocoKeypoints(**data)

            img_dict: typing.Dict[int, CocoImage] = {}
            for img in parsed_coco.images:
                img_dict[img.id] = img

            category_dict: typing.Dict[int, CocoKeypointCategory] = {}
            for category in parsed_coco.categories:
                category_dict[category.id] = category

            # iterate over all annotations and create a dict {img_id: {semantic_type : [keypoints]}}
            # make sure to deal with multiple occurances of same semantic_type in one image (e.g. multipe humans in one image)
            annotation_dict = defaultdict(
                lambda: defaultdict(lambda: [])
            )  # {img_id: {channel : [keypoints for that channel]}}
            for annotation in parsed_coco.annotations:
                # add all keypoints from this annotation to the corresponding image in the dict

                img = img_dict[annotation.image_id]
                category = category_dict[annotation.category_id]
                semantic_classes = category.keypoints

                keypoints = annotation.keypoints
                keypoints = self.split_list_in_keypoints(keypoints)
                for semantic_type, keypoint in zip(semantic_classes, keypoints):
                    annotation_dict[annotation.image_id][semantic_type].append(keypoint)

            # iterate over each image and all it's annotations
            # filter the visible keypoints
            # and group them by channel
            dataset = []
            for img_id, keypoint_dict in annotation_dict.items():
                img_channels_keypoints = [[] for _ in range(len(self.keypoint_channel_configuration))]
                for semantic_type, keypoints in keypoint_dict.items():
                    for keypoint in keypoints:

                        if (
                            min(keypoint[:2]) < 0
                            or keypoint[0] > img_dict[img_id].width
                            or keypoint[1] > img_dict[img_id].height
                        ):
                            print("keypoint outside of image, ignoring.")
                            continue
                        if self.is_keypoint_visible(keypoint):
                            channel_idx = self.get_keypoint_channel_index(semantic_type)
                            if channel_idx > -1:
                                img_channels_keypoints[channel_idx].append(keypoint[:2])

                dataset.append([img_dict[img_id].file_name, img_channels_keypoints])

            return dataset

    def get_keypoint_channel_index(self, semantic_type: str) -> int:
        """
        given a semantic type, get it's channel according to the channel configuration.
        Returns -1 if the semantic type couldn't be found.
        """

        for i, types_in_channel in enumerate(self.keypoint_channel_configuration):
            if semantic_type in types_in_channel:
                return i
        return -1

    def is_keypoint_visible(self, keypoint: COCO_KEYPOINT_TYPE) -> bool:
        """
        Args:
            keypoint (list): [u,v,flag]

        Returns:
            bool: True if current keypoint is considered visible according to the dataset configuration, else False
        """
        if self.detect_only_visible_keypoints:
            # filter out occluded keypoints with flag 1.0
            return keypoint[2] > 1.5
        else:
            # filter out non-labeled keypoints with flag 0.0
            return keypoint[2] > 0.5

    @staticmethod
    def split_list_in_keypoints(list_to_split: List[COCO_KEYPOINT_TYPE]) -> List[List[COCO_KEYPOINT_TYPE]]:
        """
        splits list [u1,v1,f1,u2,v2,f2,...] to [[u,v,f],..]
        """
        n = 3
        output = [list_to_split[i : i + n] for i in range(0, len(list_to_split), n)]
        return output

    @staticmethod
    def collate_fn(data):
        """custom collate function for use with the torch dataloader

        Note that it could have been more efficient to padd for each channel separately, but it's not worth the trouble as even
        for 100 channels with each 100 occurances the padded data size is still < 1kB..

        Args:
            data: list of tuples (image, keypoints); image = 3xHxW tensor; keypoints = List(c x list(? keypoints ))

        Returns:
            (images, keypoints); Images as a torch tensor Nx3xHxW,
            keypoints is a nested list of lists. where each item is a tensor (K,2) with K the number of keypoints
            for that channel and that sample:

                List(List(Tensor(K,2))) -> C x N x Tensor(max_keypoints_for_any_channel_in_batch x 2)

        Note there is no padding, as all values need to be unpacked again in the detector to create all the heatmaps,
        unlike e.g. NLP where you directly feed the padded sequences to the network.
        """
        images, keypoints = zip(*data)

        # convert the list of keypoints to a 2D tensor
        keypoints = [[torch.tensor(x) for x in y] for y in keypoints]
        # reorder to have the different keypoint channels as  first dimension
        # C x N x K x 2 , K = variable number of keypoints for each (N,C)
        reordered_keypoints = [[keypoints[i][j] for i in range(len(keypoints))] for j in range(len(keypoints[0]))]

        images = torch.stack(images)

        return images, reordered_keypoints
