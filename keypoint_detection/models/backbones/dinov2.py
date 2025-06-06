import timm
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Resize

from keypoint_detection.models.backbones.base_backbone import Backbone


class UpSamplingBlock(nn.Module):
    """
    A very basic Upsampling block (these params have to be learnt from scratch so keep them small)

    x --> up ---> conv1 --> norm -> relu

    """

    def __init__(self, n_channels_in, n_channels_out, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=n_channels_in,
            out_channels=n_channels_out,
            kernel_size=kernel_size,
            bias=False,
            padding="same",
        )

        self.norm1 = nn.BatchNorm2d(n_channels_out)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        # bilinear is not deterministic, use nearest neighbor instead
        x = nn.functional.interpolate(x, scale_factor=2.0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # second conv as in original UNet upsampling block decreases performance
        # probably because I was using a small dataset that did not have enough data to learn the extra parameters
        return x


class DinoV2Up(Backbone):
    """
    backbone based on a frozen Dino-v2 ViT-S  model and a number of conv-based upsampling blocks to go from patch-level to pixel-level.
    Images are resized to 518x518 before being fed to the ViT.

    The Dino v2 paper considers adding  both a linear layer  and a full-blown DPT head to the intermediate output of the last 4 blocks of the ViT.

    This model can be considered as a simpler alternative to the DPT head that also aims to increase resolution of the features.

    The upsample blocks add about 6M params, bringing the total to 28 params.
    only these blocks are trained, the dino model is frozen.

    Dinov2 paper: https://arxiv.org/pdf/2304.07193#page=13.87
    DPT paper: https://arxiv.org/abs/2103.13413


    THe head is most likely not the optimal architecture. reducing the #params in the decoder does not work for sure.
      Unfreezing the dino model doesn't work either (for small datasets).

    note that the model is trained on 518x518 images and will perform best on images of that size.
    images of other sizes are first resized to 518x518 before being fed to the model and their features are then resized back to the original size after being upsampled to 518x518.
    #TODO: figure out if it is better to change # patches for larger images, by interpolating the positional embeddings. As described in the original ViT paper
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )

        # get model specific transforms (normalization, resize)
        self.img_resizer = Resize((518, 518))  # specific to DinoV2 ViT

        self.feature_extractor = create_feature_extractor(
            self.encoder, ["blocks.8", "blocks.9", "blocks.10", "blocks.11"]
        )

        # freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.upsamplingblocks = nn.ModuleList(
            [
                UpSamplingBlock(4 * 384, 384, 3),
                UpSamplingBlock(384, 192, 3),
                UpSamplingBlock(192, 96, 3),
                UpSamplingBlock(96, 96, 3),
            ]
        )

    def forward(self, x):
        orig_image_shape = x.shape[-2:]
        x = self.img_resizer(x)
        features = self.feature_extractor(x)  # [(B,1370,384)]
        features = list(features.values())
        # concatenate the features
        features = torch.cat(features, dim=2)
        # drop class token patch
        features = features[:, 1:]  # (B, 1369, 384)

        # reshape to (B,B, 37,37,4*384)
        features = features.view(features.shape[0], 37, 37, -1)

        # permute to (B, 4*384, 37, 37)
        features = features.permute(0, 3, 1, 2)

        # upsample 3 times 2x to 37*8 = 296
        for i in range(3):
            features = self.upsamplingblocks[i](features)

        # resize to 518/2 = 259
        features = nn.functional.interpolate(features, size=(259, 259))
        # upsample final time to 518
        features = self.upsamplingblocks[-1](features)

        # now resize to original image shape
        features = nn.functional.interpolate(features, size=orig_image_shape)
        return features

    def get_n_channels_out(self):
        return 96


if __name__ == "__main__":
    model = DinoV2Up()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num trainable params = {num_params/10**6:.2f} M")

    x = torch.zeros((1, 3, 512, 512))
    y = model(x)
    print(y.shape)
