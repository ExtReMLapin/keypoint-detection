import argparse
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.metrics import DetectedKeypoint, Keypoint, KeypointAPMetrics
from keypoint_detection.utils.heatmap import BCE_loss, create_heatmap_batch, get_keypoints_from_heatmap_batch_maxpool, focal_loss_with_logits, adaptive_focal_loss_with_logits
from keypoint_detection.utils.visualization import (
    get_logging_label_from_channel_configuration,
    visualize_predicted_heatmaps,
    visualize_predicted_keypoints,
)
from keypoint_detection.models.optimized_threshold import OptimizedThresholdSearch


class KeypointDetector(pl.LightningModule):
    """
    keypoint Detector using Spatial Heatmaps.
    There can be N channels of keypoints, each with its own set of ground truth keypoints.
    The mean Average precision is used to calculate the performance.

    """

    @staticmethod
    def add_model_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        """
        parser = parent_parser.add_argument_group("KeypointDetector")

        parser.add_argument(
            "--heatmap_sigma",
            default=2,
            type=int,
            help="The size of the Gaussian blobs that are used to create the ground truth heatmaps from the keypoint coordinates.",
        )
        parser.add_argument(
            "--minimal_keypoint_extraction_pixel_distance",
            type=int,
            default=1,
            help="the minimal pixel-distance between two keypoints. Allows for some non-maximum surpression.",
        )
        parser.add_argument(
            "--maximal_gt_keypoint_pixel_distances",
            type=str,
            default="2 4",
            help="The treshold distance(s) for the AP metric to consider a detection as a True Positive. Separate multiple values by a space to compute the AP for all values.",
        )
        parser.add_argument("--learning_rate", type=float, default=3e-4)  # Karpathy constant
        parser.add_argument(
            "--ap_epoch_start",
            type=int,
            default=1,
            help="Epoch at which to start calculating the AP every `ap_epoch_frequency` epochs.",
        )
        parser.add_argument(
            "--ap_epoch_freq",
            type=int,
            default=2,
            help="Rate at which to calculate the AP metric if epoch > `ap_epoch_start`",
        )
        parser.add_argument(
            "--lr_scheduler_relative_threshold",
            default=0.0,
            type=float,
            help="relative threshold for the OnPlateauLRScheduler. If the training epoch loss does not decrease with this fraction for 2 consective epochs, lr is decreased with factor 10.",
        )
        parser.add_argument(
            "--max_keypoints",
            default=800,
            type=int,
            help="the maximum number of keypoints to predict from the generated heatmaps. If set to -1, skimage will look for all peaks in the heatmap, if set to N (N>0) it will return the N most most certain ones.",
        )
        
        parser.add_argument(
            "--lr_scheduler", 
            type=str, 
            choices=["none", "cosine", "onecycle", "step", "plateau"],
            default="cosine",
            help="Learning rate scheduler type"
        )
        parser.add_argument(
            "--lr_warmup_epochs", 
            type=int, 
            default=10,
            help="Number of warmup epochs for cosine/onecycle schedulers"
        )
        parser.add_argument(
            "--lr_step_milestones", 
            type=str, 
            default="150,250,350",
            help="Comma-separated epochs for step scheduler"
        )
        parser.add_argument(
            "--lr_step_gamma", 
            type=float, 
            default=0.1,
            help="LR decay factor for step scheduler"
        )
        
        parent_parser = KeypointDetector.add_threshold_optimization_args(parent_parser)
        return parent_parser

    def __init__(
        self,
        heatmap_sigma: int,
        maximal_gt_keypoint_pixel_distances: str,
        minimal_keypoint_extraction_pixel_distance: int,
        learning_rate: float,
        backbone: Backbone,
        keypoint_channel_configuration: List[List[str]],
        ap_epoch_start: int,
        ap_epoch_freq: int,
        lr_scheduler_relative_threshold: float,
        max_keypoints: int,
        enable_threshold_optimization:bool,
        threshold_search_low:float,
        threshold_search_high:float,
        threshold_search_tolerance:float,
        threshold_optimization_method: str = "optimized",
        threshold_coarse_steps: int = 20,
        threshold_fine_steps: int = 10,
        threshold_batch_size: int = 16,
        use_focal_loss: bool = False,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        use_adaptive_focal_loss: bool = False,
        **kwargs,
    ):
        """[summary]

        Args:
            see argparse help strings for documentation.

            kwargs: Pythonic catch for the other named arguments, used so that we can use a dict with ALL system hyperparameters to initialise the model from this
                    hyperparamater configuration dict. The alternative is to add a single 'hparams' argument to the init function, but this is imo less readable.
                    cf https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html for an overview.
        """
        super().__init__()
        ## No need to manage devices ourselves, pytorch.lightning does all of that.
        ## device can be accessed through self.device if required.

        # to add new hyperparameters:
        # 1. define as named arg in the init (and use them)
        # 2. add to the argparse method of this module
        # 3. pass them along when calling the train.py file to override their default value
        self.lr_scheduler = kwargs.get('lr_scheduler', 'cosine')
        self.lr_warmup_epochs = kwargs.get('lr_warmup_epochs', 10)
        self.lr_step_milestones = kwargs.get('lr_step_milestones', '150,250,350')
        self.lr_step_gamma = kwargs.get('lr_step_gamma', 0.1)
        self.learning_rate = learning_rate
        self.heatmap_sigma = heatmap_sigma
        self.ap_epoch_start = ap_epoch_start
        self.ap_epoch_freq = ap_epoch_freq
        self.minimal_keypoint_pixel_distance = minimal_keypoint_extraction_pixel_distance
        self.lr_scheduler_relative_threshold = lr_scheduler_relative_threshold
        self.max_keypoints = max_keypoints
        self.keypoint_channel_configuration = keypoint_channel_configuration
        # parse the gt pixel distances
        if isinstance(maximal_gt_keypoint_pixel_distances, str):
            maximal_gt_keypoint_pixel_distances = [
                int(val) for val in maximal_gt_keypoint_pixel_distances.strip().split(" ")
            ]
        self.maximal_gt_keypoint_pixel_distances = maximal_gt_keypoint_pixel_distances

        self.ap_training_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) for _ in self.keypoint_channel_configuration
        ]
        self.ap_validation_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) for _ in self.keypoint_channel_configuration
        ]

        self.ap_test_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) for _ in self.keypoint_channel_configuration
        ]

        self.n_heatmaps = len(self.keypoint_channel_configuration)

        head = nn.Conv2d(
            in_channels=backbone.get_n_channels_out(),
            out_channels=self.n_heatmaps,
            kernel_size=(3, 3),
            padding="same",
        )

        # expect output of backbone to be normalized!
        # so by filling bias to -4, the sigmoid should be on avg sigmoid(-4) =  0.02
        # which is consistent with the desired heatmaps that are zero almost everywhere.
        # setting too low would result in loss of gradients..
        head.bias.data.fill_(-4)

        self.unnormalized_model = nn.Sequential(
            backbone,
            head,
        )  # NO sigmoid to combine it in the loss! (needed for FP16)

        # save hyperparameters to logger, to make sure the model hparams are saved even if
        # they are not included in the config (i.e. if they are kept at the defaults).
        # this is for later reference (e.g. checkpoint loading) and consistency.
        

        self._most_recent_val_mean_ap = 0.0  # used to store the most recent validation mean AP and log it in each epoch, so that checkpoint can be chosen based on this one.

        # For storing validation data during epoch
        self.validation_heatmaps = []
        self.validation_gt_keypoints = []
        
        # Optimal threshold found during validation
        self.optimal_threshold = 0.01  # Default threshold
        self.best_validation_map = 0.0
        
        # Threshold optimization parameters (can be made configurable via argparse)
        self.enable_threshold_optimization = enable_threshold_optimization
        self.threshold_search_low = threshold_search_low
        self.threshold_search_high = threshold_search_high
        self.threshold_search_tolerance = threshold_search_tolerance
        self.threshold_optimization_method = threshold_optimization_method
        self.threshold_coarse_steps = threshold_coarse_steps
        self.threshold_fine_steps = threshold_fine_steps
        self.threshold_batch_size = threshold_batch_size
        self.use_focal_loss = use_focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.use_adaptive_focal_loss = use_adaptive_focal_loss
        self.save_hyperparameters(ignore=["**kwargs", "backbone"])

    def forward(self, x: torch.Tensor):
        """
        x shape must be of shape (N,3,H,W)
        returns tensor with shape (N, n_heatmaps, H,W)
        """
        return torch.sigmoid(self.forward_unnormalized(x))

    def forward_unnormalized(self, x: torch.Tensor):
        return self.unnormalized_model(x)

    def configure_optimizers(self):
        """
        Configures an Adam optimizer.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)

        if self.lr_scheduler == "none":
                return {"optimizer": self.optimizer}
            
        elif self.lr_scheduler == "cosine":
            # Cosine annealing - excellent for long training runs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.trainer.max_epochs - self.lr_warmup_epochs,
                eta_min=self.learning_rate * 0.01  # End at 1% of initial LR
            )
            
            if self.lr_warmup_epochs > 0:
                # Add warmup
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,  # Start at 10% of LR
                    total_iters=self.lr_warmup_epochs
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[self.lr_warmup_epochs]
                )
            
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        
        elif self.lr_scheduler == "onecycle":
            # OneCycleLR - very effective for training from scratch
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate * 3,  # Peak at 3x base LR
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
                pct_start=0.3,  # 30% warmup, 70% decay
                anneal_strategy='cos'
            )
            
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"  # OneCycle needs step-wise updates
                }
            }
        
        elif self.lr_scheduler == "step":
            # Step decay at specific epochs
            milestones = [int(x) for x in self.lr_step_milestones.split(",")]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=self.lr_step_gamma
            )
            
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        
        elif self.lr_scheduler == "plateau":
            # Reduce on plateau - good for validation-driven training
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=15,  # More patience for small datasets
                threshold=0.01,
                threshold_mode='rel',
                verbose=True,
                min_lr=self.learning_rate * 0.01
            )
            
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "validation/epoch_loss",  # Monitor validation loss
                    "interval": "epoch"
                }
            }

    def shared_step(self, batch, batch_idx, include_visualization_data_in_result_dict=False) -> Dict[str, Any]:
        """
        shared step for train and validation step that computes the heatmaps and losses and
        creates a result dict for later use in the train, validate and test step.

        batch: img, keypoints
        where img is a Nx3xHxW tensor
        and keypoints a nested list of len(channels) x N with K_ij x 2 tensors containing the keypoints for each channel and each sample in the batch

        returns:

        shared_dict (Dict): a dict with a.o. heatmaps, gt_keypoints and losses
        """
        input_images, keypoint_channels = batch
        heatmap_shape = input_images[0].shape[1:]

        gt_heatmaps = [
            create_heatmap_batch(heatmap_shape, keypoint_channel, self.heatmap_sigma, self.device)
            for keypoint_channel in keypoint_channels
        ]

        input_images = input_images.to(self.device)

        ## predict and compute losses
        predicted_unnormalized_maps = self.forward_unnormalized(input_images)
        predicted_heatmaps = torch.sigmoid(predicted_unnormalized_maps)
        channel_losses = []
        channel_gt_losses = []

        result_dict = {}
        for channel_idx in range(len(self.keypoint_channel_configuration)):
            if self.use_focal_loss:
                if self.use_adaptive_focal_loss:
                    loss_fn = adaptive_focal_loss_with_logits
                else:
                    loss_fn = focal_loss_with_logits
                
                channel_losses.append(
                    loss_fn(
                        predicted_unnormalized_maps[:, channel_idx, :, :], 
                        gt_heatmaps[channel_idx],
                        alpha=self.focal_loss_alpha,
                        gamma=self.focal_loss_gamma
                    )
                )
            else:
                # Original BCE loss
                channel_losses.append(
                    nn.functional.binary_cross_entropy_with_logits(
                        predicted_unnormalized_maps[:, channel_idx, :, :], 
                        gt_heatmaps[channel_idx]
                    )
                )
            
            with torch.no_grad():
                channel_gt_losses.append(BCE_loss(gt_heatmaps[channel_idx], gt_heatmaps[channel_idx]))

            # pass losses and other info to result dict
            result_dict.update(
                {f"{self.keypoint_channel_configuration[channel_idx]}_loss": channel_losses[channel_idx].detach()}
            )

        loss = sum(channel_losses)
        gt_loss = sum(channel_gt_losses)
        result_dict.update({"loss": loss, "gt_loss": gt_loss})

        if include_visualization_data_in_result_dict:
            result_dict.update(
                {
                    "input_images": input_images.detach().cpu(),
                    "gt_keypoints": keypoint_channels,
                    "predicted_heatmaps": predicted_heatmaps.detach().cpu(),
                    "gt_heatmaps": gt_heatmaps,
                }
            )

        return result_dict

    def training_step(self, train_batch, batch_idx):
        log_images = batch_idx == 0 and self.current_epoch > 0 and self.is_ap_epoch()
        should_log_ap = self.is_ap_epoch() and batch_idx < 20  # limit AP calculation to first 20 batches to save time
        include_vis_data = log_images or should_log_ap

        result_dict = self.shared_step(
            train_batch, batch_idx, include_visualization_data_in_result_dict=include_vis_data
        )

        if should_log_ap:
            self.update_ap_metrics(result_dict, self.ap_training_metrics)

        if log_images:
            image_grids = self.visualize_predictions_channels(result_dict)
            self.log_channel_predictions_grids(image_grids, mode="train")

        for channel_name in self.keypoint_channel_configuration:
            self.log(f"train/{channel_name}", result_dict[f"{channel_name}_loss"], sync_dist=True)

        # self.log("train/loss", result_dict["loss"])
        self.log("train/gt_loss", result_dict["gt_loss"], sync_dist=True)
        self.log("train/loss", result_dict["loss"], on_epoch=True, sync_dist=True)  # also logs steps?
        return result_dict

    def update_ap_metrics(self, result_dict, ap_metrics):
        predicted_heatmaps = result_dict["predicted_heatmaps"]
        gt_keypoints = result_dict["gt_keypoints"]
        for channel_idx in range(len(self.keypoint_channel_configuration)):
            predicted_heatmaps_channel = predicted_heatmaps[:, channel_idx, :, :]
            gt_keypoints_channel = gt_keypoints[channel_idx]
            self.update_channel_ap_metrics(predicted_heatmaps_channel, gt_keypoints_channel, ap_metrics[channel_idx])

    def visualize_predictions_channels(self, result_dict):
        input_images = result_dict["input_images"]
        gt_heatmaps = result_dict["gt_heatmaps"]
        predicted_heatmaps = result_dict["predicted_heatmaps"]

        image_grids = []
        for channel_idx in range(len(self.keypoint_channel_configuration)):
            grid = visualize_predicted_heatmaps(
                input_images,
                predicted_heatmaps[:, channel_idx, :, :],
                gt_heatmaps[channel_idx].cpu(),
            )
            image_grids.append(grid)
        return image_grids

    def log_channel_predictions_grids(self, image_grids, mode: str):
        if self.trainer.is_global_zero:
            for channel_configuration, grid in zip(self.keypoint_channel_configuration, image_grids):
                label = get_logging_label_from_channel_configuration(channel_configuration, mode)
                image_caption = "top: predicted heatmaps, bottom: gt heatmaps"
                self.logger.experiment.log({label: wandb.Image(grid, caption=image_caption, file_type="jpg")})

    def visualize_predicted_keypoints(self, result_dict):
        images = result_dict["input_images"]
        predicted_heatmaps = result_dict["predicted_heatmaps"]
        # get the keypoints from the heatmaps
        predicted_heatmaps = predicted_heatmaps.detach().float()
        predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(
            predicted_heatmaps, self.max_keypoints, self.minimal_keypoint_pixel_distance, abs_max_threshold=0.005
        )
        # overlay the images with the keypoints
        grid = visualize_predicted_keypoints(images, predicted_keypoints, self.keypoint_channel_configuration)
        return grid

    def log_predicted_keypoints(self, grid, mode=str):
        if self.trainer.is_global_zero:
            label = f"predicted_keypoints_{mode}"
            image_caption = "predicted keypoints"
            self.logger.experiment.log({label: wandb.Image(grid, caption=image_caption)})

    def validation_step(self, val_batch, batch_idx):
        # Your existing validation_step code, but add data collection
        result_dict = self.shared_step(val_batch, batch_idx, include_visualization_data_in_result_dict=True)
        
        if self.is_ap_epoch() and self.enable_threshold_optimization:
            # Store data for threshold optimization
            predicted_heatmaps = result_dict["predicted_heatmaps"].detach().cpu()
            gt_keypoints = result_dict["gt_keypoints"]
            
            # Store each sample in the batch separately for easier indexing later
            for sample_idx in range(predicted_heatmaps.shape[0]):
                self.validation_heatmaps.append(predicted_heatmaps[sample_idx:sample_idx+1])
                # Reorganize gt_keypoints: from [channel][batch_sample] to [sample][channel]
                sample_gt_keypoints = []
                for channel_idx in range(len(gt_keypoints)):
                    if sample_idx < len(gt_keypoints[channel_idx]):
                        sample_gt_keypoints.append(gt_keypoints[channel_idx][sample_idx])
                    else:
                        sample_gt_keypoints.append(torch.tensor([]))
                self.validation_gt_keypoints.append(sample_gt_keypoints)
            
            log_images = batch_idx == 0 and self.current_epoch > 0
            if log_images and isinstance(self.logger, pl.loggers.wandb.WandbLogger):
                channel_grids = self.visualize_predictions_channels(result_dict)
                self.log_channel_predictions_grids(channel_grids, mode="validation")
                
                keypoint_grids = self.visualize_predicted_keypoints(result_dict)
                self.log_predicted_keypoints(keypoint_grids, mode="validation")
        
        elif self.is_ap_epoch() and not self.enable_threshold_optimization:
            # Use old method if threshold optimization is disabled
            self.update_ap_metrics(result_dict, self.ap_validation_metrics)
            keypoint_grids = self.visualize_predicted_keypoints(result_dict)
            self.log_predicted_keypoints(keypoint_grids, mode="validation")

        self.log("validation/epoch_loss", result_dict["loss"], sync_dist=True)
        self.log("validation/gt_loss", result_dict["gt_loss"], sync_dist=True)


    def ternary_search_optimal_threshold(
        self, 
        heatmaps: List[torch.Tensor], 
        gt_keypoints: List[List], 
        low: float = None, 
        high: float = None, 
        tolerance: float = None
    ) -> Tuple[float, float]:
        """
        Use ternary search to find the threshold that maximizes mAP.
        
        Args:
            heatmaps: List of validation heatmaps (one per sample)
            gt_keypoints: List of GT keypoints per sample [sample][channel][keypoint_list]
            low: Lower bound for threshold search
            high: Upper bound for threshold search  
            tolerance: Search tolerance
            
        Returns:
            (optimal_threshold, best_map)
        """
        
        # Use instance parameters if not provided
        if low is None:
            low = self.threshold_search_low
        if high is None:
            high = self.threshold_search_high
        if tolerance is None:
            tolerance = self.threshold_search_tolerance
        
        print(f"# Searching optimal threshold in range [{low:.4f}, {high:.4f}] with tolerance {tolerance}")
        
        def evaluate_threshold(threshold: float) -> float:
            """Evaluate mAP for a given threshold"""
            # Create fresh metrics for this threshold evaluation
            temp_metrics = [
                KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) 
                for _ in self.keypoint_channel_configuration
            ]
            
            # Extract keypoints with this threshold for each sample
            for sample_idx, (sample_heatmap, sample_gt_keypoints) in enumerate(zip(heatmaps, gt_keypoints)):
                
                keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
                    sample_heatmap.float(),
                    self.max_keypoints,
                    self.minimal_keypoint_pixel_distance,
                    abs_max_threshold=threshold,
                    return_scores=True
                )
                
                # Update metrics for each channel
                for channel_idx in range(len(self.keypoint_channel_configuration)):
                    # Detected keypoints for this channel
                    detected_keypoints_channel = []
                    if (channel_idx < len(keypoints[0]) and 
                        len(keypoints[0][channel_idx]) > 0 and 
                        len(scores[0][channel_idx]) > 0):
                        detected_keypoints_channel = [
                            DetectedKeypoint(int(kp[0]), int(kp[1]), float(score)) 
                            for kp, score in zip(keypoints[0][channel_idx], scores[0][channel_idx])
                        ]
                    
                    # Ground truth keypoints for this channel
                    gt_keypoints_channel = []
                    if (channel_idx < len(sample_gt_keypoints) and 
                        len(sample_gt_keypoints[channel_idx]) > 0):
                        gt_keypoints_channel = [
                            Keypoint(int(k[0]), int(k[1])) 
                            for k in sample_gt_keypoints[channel_idx]
                        ]
                    
                    temp_metrics[channel_idx].update(detected_keypoints_channel, gt_keypoints_channel)
            
            # Compute mean AP across all channels and thresholds
            total_ap = 0.0
            total_count = 0
            
            for channel_metrics in temp_metrics:
                channel_aps = channel_metrics.compute()
                if len(channel_aps) > 0:
                    total_ap += sum(channel_aps.values())
                    total_count += len(channel_aps)
                channel_metrics.reset()
            
            mean_ap = total_ap / max(total_count, 1)
            return mean_ap
        
        # Ternary search implementation
        iteration = 0
        max_iterations = 50  # Safety limit
        
        while high - low > tolerance and iteration < max_iterations:
            mid1 = low + (high - low) / 3
            mid2 = high - (high - low) / 3
            
            map1 = evaluate_threshold(mid1)
            map2 = evaluate_threshold(mid2)
            
            if iteration % 5 == 0:  # Log progress every 5 iterations
                print(f"# Iteration {iteration}: [{low:.4f}, {high:.4f}] -> mAP({mid1:.4f})={map1:.4f}, mAP({mid2:.4f})={map2:.4f}")
            
            if map1 < map2:
                low = mid1
            else:
                high = mid2
                
            iteration += 1
        
        optimal_threshold = (low + high) / 2
        best_map = evaluate_threshold(optimal_threshold)
        
        print(f"# Search completed in {iteration} iterations")
        
        return optimal_threshold, best_map


    def test_step(self, test_batch, batch_idx):
        # no need to switch model to eval mode, this is handled by pytorch lightning
        result_dict = self.shared_step(test_batch, batch_idx, include_visualization_data_in_result_dict=True)
        self.update_ap_metrics(result_dict, self.ap_test_metrics)
        # only log first 10 batches to reduce storage space
        if batch_idx < 10 and isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            image_grids = self.visualize_predictions_channels(result_dict)
            self.log_channel_predictions_grids(image_grids, mode="test")

            keypoint_grids = self.visualize_predicted_keypoints(result_dict)
            self.log_predicted_keypoints(keypoint_grids, mode="test")

        self.log("test/epoch_loss", result_dict["loss"])
        self.log("test/gt_loss", result_dict["gt_loss"])

    def log_and_reset_mean_ap(self, mode: str):
        mean_ap_per_threshold = torch.zeros(len(self.maximal_gt_keypoint_pixel_distances),  device=self.device)
        if mode == "train":
            metrics = self.ap_training_metrics
        elif mode == "validation":
            metrics = self.ap_validation_metrics
        elif mode == "test":
            metrics = self.ap_test_metrics
        else:
            raise ValueError(f"mode {mode} not recognized")

        # calculate APs for each channel and each threshold distance, and log them
        print(f" # {mode} metrics:")
        for channel_idx, channel_name in enumerate(self.keypoint_channel_configuration):
            channel_aps = self.compute_and_log_metrics_for_channel(metrics[channel_idx], channel_name, mode)
            mean_ap_per_threshold += torch.tensor(channel_aps, device=self.device)

        # calculate the mAP over all channels for each threshold distance, and log them
        for i, maximal_distance in enumerate(self.maximal_gt_keypoint_pixel_distances):
            self.log(
                f"{mode}/meanAP/d={float(maximal_distance):.1f}",
                mean_ap_per_threshold[i] / len(self.keypoint_channel_configuration),
                sync_dist=True,
            )

        # calculate the mAP over all channels and all threshold distances, and log it
        mean_ap = mean_ap_per_threshold.mean() / len(self.keypoint_channel_configuration)
        self.log(f"{mode}/meanAP", mean_ap, sync_dist=True)
        self.log(f"{mode}/meanAP/meanAP", mean_ap, sync_dist=True)

        if mode == "validation":
            self._most_recent_val_mean_ap = mean_ap

    def on_train_epoch_end(self):
        """
        Called on the end of a training epoch.
        Used to compute and log the AP metrics.
        """
        if self.is_ap_epoch():
            self.log_and_reset_mean_ap("train")

    # Replace the ternary_search_optimal_threshold method with this optimized version
    def optimized_threshold_search(
        self, 
        heatmaps: List[torch.Tensor], 
        gt_keypoints: List[List], 
        low: float = None, 
        high: float = None
    ) -> Tuple[float, float]:
        """
        Optimized threshold search that's ~10-50x faster than ternary search
        
        Args:
            heatmaps: List of validation heatmaps (one per sample)
            gt_keypoints: List of GT keypoints per sample [sample][channel][keypoint_list]
            low: Lower bound for threshold search
            high: Upper bound for threshold search
            
        Returns:
            (optimal_threshold, best_map)
        """
        
        # Use instance parameters if not provided
        if low is None:
            low = self.threshold_search_low
        if high is None:
            high = self.threshold_search_high
        
        print(f"# Optimized threshold search for {len(heatmaps)} validation samples")
        
        # Create optimizer
        optimizer = OptimizedThresholdSearch(
            self.maximal_gt_keypoint_pixel_distances,
            self.max_keypoints,
            self.minimal_keypoint_pixel_distance
        )
        
        # Pre-extract keypoints once (this is the key optimization)
        optimizer.extract_all_keypoints_once(heatmaps, gt_keypoints)
        
        # Find optimal threshold using two-stage grid search
        optimal_threshold, best_map = optimizer.grid_search_optimal_threshold(
            low, high, 
            num_coarse_steps=20,  # Adjust based on speed/accuracy tradeoff
            num_fine_steps=10
        )
        
        return optimal_threshold, best_map

    # Updated on_validation_epoch_end method
    def on_validation_epoch_end(self):
        """
        Optimized validation_epoch_end with faster threshold optimization
        """
        if self.is_ap_epoch():
            if self.enable_threshold_optimization and len(self.validation_heatmaps) > 0:
                print(f"\n# Optimizing confidence threshold for epoch {self.current_epoch}")
                print(f"# Using method: {self.threshold_optimization_method}")
                
                # Choose optimization method
                if self.threshold_optimization_method == "optimized":
                    optimal_threshold, best_map = self.optimized_threshold_search(
                        self.validation_heatmaps, 
                        self.validation_gt_keypoints
                    )
                elif self.threshold_optimization_method == "memory_efficient":
                    optimal_threshold, best_map = self.memory_efficient_threshold_search(
                        self.validation_heatmaps, 
                        self.validation_gt_keypoints
                    )
                elif self.threshold_optimization_method == "ternary":
                    # Fallback to original slow method
                    optimal_threshold, best_map = self.ternary_search_optimal_threshold(
                        self.validation_heatmaps, 
                        self.validation_gt_keypoints
                    )
                else:
                    raise ValueError(f"Unknown threshold optimization method: {self.threshold_optimization_method}")
                
                print(f"# Optimal threshold found: {optimal_threshold:.6f} (mAP: {best_map:.4f})")
                
                # Update stored optimal threshold
                self.optimal_threshold = optimal_threshold
                self.best_validation_map = best_map
                
                # Log the optimal threshold
                self.log("validation/optimal_threshold", optimal_threshold, sync_dist=True)
                self.log("validation/threshold_optimized_mAP", best_map, sync_dist=True)
                
                # Re-evaluate with optimal threshold for detailed per-channel metrics
                # This is much faster now since we use the pre-extracted keypoints
                self.ap_validation_metrics = [
                    KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) 
                    for _ in self.keypoint_channel_configuration
                ]
                
                # Fast re-evaluation using cached keypoints
                optimizer = OptimizedThresholdSearch(
                    self.maximal_gt_keypoint_pixel_distances,
                    self.max_keypoints,
                    self.minimal_keypoint_pixel_distance
                )
                optimizer.extract_all_keypoints_once(self.validation_heatmaps, self.validation_gt_keypoints)
                
                # Get detailed metrics for logging
                for sample_idx in range(len(self.validation_heatmaps)):
                    sample_heatmap = self.validation_heatmaps[sample_idx]
                    sample_gt_keypoints = self.validation_gt_keypoints[sample_idx]
                    
                    # Use pre-extracted keypoints and filter by optimal threshold
                    sample_keypoints = optimizer.cached_keypoints[sample_idx]
                    sample_scores = optimizer.cached_scores[sample_idx]
                    
                    for channel_idx in range(len(self.keypoint_channel_configuration)):
                        # Fast threshold filtering
                        detected_keypoints_channel = []
                        if (channel_idx < len(sample_keypoints) and 
                            len(sample_keypoints[channel_idx]) > 0 and 
                            len(sample_scores[channel_idx]) > 0):
                            
                            detected_keypoints_channel = [
                                DetectedKeypoint(int(kp[0]), int(kp[1]), float(score))
                                for kp, score in zip(sample_keypoints[channel_idx], sample_scores[channel_idx])
                                if score > optimal_threshold
                            ]
                        
                        # Ground truth keypoints for this channel
                        gt_keypoints_channel = []
                        if (channel_idx < len(sample_gt_keypoints) and 
                            len(sample_gt_keypoints[channel_idx]) > 0):
                            gt_keypoints_channel = [
                                Keypoint(int(k[0]), int(k[1])) 
                                for k in sample_gt_keypoints[channel_idx]
                            ]
                        
                        self.ap_validation_metrics[channel_idx].update(detected_keypoints_channel, gt_keypoints_channel)
                
                # Clear stored validation data
                self.validation_heatmaps.clear()
                self.validation_gt_keypoints.clear()
            
            #elif self.is_ap_epoch() and not self.enable_threshold_optimization:
            #    # Use old method if threshold optimization is disabled
            #    self.update_ap_metrics(result_dict, self.ap_validation_metrics)

        # Log final metrics (works for both optimized and non-optimized cases)
        self.log_and_reset_mean_ap("validation")

        # Update checkpointing metric
        self.log("checkpointing_metrics/valmeanAP", self._most_recent_val_mean_ap, sync_dist=True)

    # For very large datasets, use memory-efficient version
    def memory_efficient_threshold_search(
        self, 
        heatmaps: List[torch.Tensor], 
        gt_keypoints: List[List], 
        low: float = None, 
        high: float = None
    ) -> Tuple[float, float]:
        """
        Memory-efficient version for very large validation sets
        Uses less memory but still much faster than original ternary search
        """
        from keypoint_detection.models.optimized_threshold import MemoryEfficientThresholdSearch
        
        if low is None:
            low = self.threshold_search_low
        if high is None:
            high = self.threshold_search_high
        
        optimizer = MemoryEfficientThresholdSearch(
            self.maximal_gt_keypoint_pixel_distances,
            self.max_keypoints,
            self.minimal_keypoint_pixel_distance
        )
        
        optimizer.set_validation_data(heatmaps, gt_keypoints)
        
        # Use coarser grid for memory efficiency
        thresholds = np.linspace(low, high, 15)
        maps = []
        
        for threshold in thresholds:
            map_score = optimizer.evaluate_threshold_on_demand(threshold)
            maps.append(map_score)
        
        best_idx = np.argmax(maps)
        optimal_threshold = thresholds[best_idx]
        best_map = maps[best_idx]
        
        return optimal_threshold, best_map

    def on_test_epoch_end(self):
        """
        Called on the end of a test epoch.
        Used to compute and log the AP metrics.
        """
        self.log_and_reset_mean_ap("test")

    def update_channel_ap_metrics(
        self, predicted_heatmaps: torch.Tensor, gt_keypoints: List[torch.Tensor], validation_metric: KeypointAPMetrics
    ):
        """
        Updates the AP metric for a batch of heatmaps and keypoins of a single channel (!)
        This is done by extracting the detected keypoints for each heatmap and combining them with the gt keypoints for the same frame, so that
        the confusion matrix can be determined together with the distance thresholds.

        predicted_heatmaps: N x H x W tensor with the batch of predicted heatmaps for a single channel
        gt_keypoints: List of size N, containing K_i x 2 tensors with the ground truth keypoints for the channel of that sample
        """

        # log corner keypoints to AP metrics for all images in this batch
        formatted_gt_keypoints = [
            [Keypoint(int(k[0]), int(k[1])) for k in frame_gt_keypoints] for frame_gt_keypoints in gt_keypoints
        ]
        batch_detected_channel_keypoints = self.extract_detected_keypoints_from_heatmap(
            predicted_heatmaps.unsqueeze(1)
        )
        batch_detected_channel_keypoints = [batch_detected_channel_keypoints[i][0] for i in range(len(gt_keypoints))]
        for i, detected_channel_keypoints in enumerate(batch_detected_channel_keypoints):
            validation_metric.update(detected_channel_keypoints, formatted_gt_keypoints[i])

    def compute_and_log_metrics_for_channel(
        self, metrics: KeypointAPMetrics, channel: str, training_mode: str
    ) -> List[float]:
        """
        logs AP of predictions of single Channel for each threshold distance.
        Also resets metric and returns resulting AP for all distances.
        """
        ap_metrics = metrics.compute()
        rounded_ap_metrics = {k: round(v, 3) for k, v in ap_metrics.items()}
        print(f"{channel} : {rounded_ap_metrics}")
        for maximal_distance, ap in ap_metrics.items():
            self.log(f"{training_mode}/{channel}_ap/d={float(maximal_distance):.1f}", ap, sync_dist=True)

        mean_ap = sum(ap_metrics.values()) / len(ap_metrics.values())
        mean_ap = torch.tensor(mean_ap, device=self.device)
        self.log(f"{training_mode}/{channel}_ap/meanAP", mean_ap, sync_dist=True)  # log top level for wandb hyperparam chart.

        metrics.reset()
        return list(ap_metrics.values())

    def is_ap_epoch(self) -> bool:
        """Returns True if the AP should be calculated in this epoch."""
        is_epch = self.ap_epoch_start <= self.current_epoch and self.current_epoch % self.ap_epoch_freq == 0
        # always log the AP in the last epoch
        is_epch = is_epch or self.current_epoch == self.trainer.max_epochs - 1

        # if user manually specified a validation frequency, we should always log the AP in that epoch
        # is_epch = is_epch or (self.current_epoch > 0 and self.trainer.check_val_every_n_epoch > 1)
        return is_epch

    def extract_detected_keypoints_from_heatmap(self, heatmap: torch.Tensor) -> List[DetectedKeypoint]:
        """
        Modified to use optimal threshold found during validation
        """
        if heatmap.dtype == torch.float16:
            heatmap_to_extract_from = heatmap.float()
        else:
            heatmap_to_extract_from = heatmap

        # Use optimal threshold if available, otherwise use default
        threshold = getattr(self, 'optimal_threshold', 0.01)
        
        keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
            heatmap_to_extract_from, 
            self.max_keypoints, 
            self.minimal_keypoint_pixel_distance, 
            abs_max_threshold=threshold,
            return_scores=True
        )
        
        detected_keypoints = [
            [[] for _ in range(heatmap_to_extract_from.shape[1])] 
            for _ in range(heatmap_to_extract_from.shape[0])
        ]
        
        for batch_idx in range(len(detected_keypoints)):
            for channel_idx in range(len(detected_keypoints[batch_idx])):
                for kp_idx in range(len(keypoints[batch_idx][channel_idx])):
                    detected_keypoints[batch_idx][channel_idx].append(
                        DetectedKeypoint(
                            keypoints[batch_idx][channel_idx][kp_idx][0],
                            keypoints[batch_idx][channel_idx][kp_idx][1],
                            scores[batch_idx][channel_idx][kp_idx],
                        )
                    )

        return detected_keypoints
    
    # Add to hyperparameters saving
    def on_save_checkpoint(self, checkpoint):
        """
        Save optimal threshold and related metrics in checkpoint
        """
        checkpoint['optimal_threshold'] = self.optimal_threshold
        checkpoint['best_validation_map'] = self.best_validation_map
        
    def on_load_checkpoint(self, checkpoint):
        """
        Load optimal threshold from checkpoint
        """
        if 'optimal_threshold' in checkpoint:
            self.optimal_threshold = checkpoint['optimal_threshold']
            print(f"# Loaded optimal threshold from checkpoint: {self.optimal_threshold:.6f}")
        if 'best_validation_map' in checkpoint:
            self.best_validation_map = checkpoint['best_validation_map']

    @staticmethod
    def add_threshold_optimization_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add threshold optimization arguments to argparse
        Call this in your add_model_argparse_args method
        """
        parser = parent_parser.add_argument_group("ThresholdOptimization")
        
        # Core threshold optimization settings
        parser.add_argument(
            "--enable_threshold_optimization",
            action="store_true",
            default=False,
            help="Enable dynamic threshold optimization during validation"
        )
        parser.add_argument(
            "--disable_threshold_optimization",
            action="store_false",
            dest="enable_threshold_optimization", 
            help="Disable dynamic threshold optimization"
        )
        
        # Search range settings
        parser.add_argument(
            "--threshold_search_low",
            type=float,
            default=0.0001,
            help="Lower bound for threshold search"
        )
        parser.add_argument(
            "--threshold_search_high", 
            type=float,
            default=0.9,
            help="Upper bound for threshold search"
        )
        
        # Optimization method selection
        parser.add_argument(
            "--threshold_optimization_method",
            type=str,
            choices=["optimized", "memory_efficient", "ternary"],
            default="optimized",
            help="Method for threshold optimization: optimized (fastest), memory_efficient (large datasets), ternary (original slow method)"
        )
        
        # Grid search parameters
        parser.add_argument(
            "--threshold_coarse_steps",
            type=int,
            default=20,
            help="Number of coarse grid search steps"
        )
        parser.add_argument(
            "--threshold_fine_steps",
            type=int,
            default=10,
            help="Number of fine grid search steps"
        )
        
        # Memory management
        parser.add_argument(
            "--threshold_batch_size",
            type=int,
            default=16,
            help="Batch size for keypoint extraction during threshold optimization"
        )
        
        # Legacy ternary search settings (kept for compatibility)
        parser.add_argument(
            "--threshold_search_tolerance",
            type=float,
            default=1e-4,
            help="Tolerance for ternary search convergence (only used with --threshold_optimization_method ternary)"
        )
        
        parser.add_argument(
            "--use_focal_loss",
            action="store_true",
            default=False,
            help="Use focal loss instead of binary cross-entropy. Recommended for imbalanced datasets like minutiae detection."
        )
        parser.add_argument(
            "--focal_loss_alpha",
            type=float,
            default=0.25,
            help="Alpha parameter for focal loss. Controls class weighting. 0.25 means 25%% weight for positive class."
        )
        parser.add_argument(
            "--focal_loss_gamma",
            type=float,
            default=2.0,
            help="Gamma parameter for focal loss. Controls focusing on hard examples. Higher values focus more on hard examples."
        )
        parser.add_argument(
            "--use_adaptive_focal_loss",
            action="store_true",
            default=False,
            help="Use adaptive focal loss that adjusts alpha based on batch statistics."
        )
        return parent_parser
        