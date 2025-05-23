import torch
import numpy as np
from typing import List, Tuple, Dict
from keypoint_detection.models.metrics import DetectedKeypoint, Keypoint, KeypointAPMetrics
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool

class OptimizedThresholdSearch:
    """Optimized threshold search for mAP maximization"""
    
    def __init__(self, 
                 maximal_gt_keypoint_pixel_distances: List[int],
                 max_keypoints: int = 800,
                 minimal_keypoint_pixel_distance: int = 1):
        self.maximal_gt_keypoint_pixel_distances = maximal_gt_keypoint_pixel_distances
        self.max_keypoints = max_keypoints
        self.minimal_keypoint_pixel_distance = minimal_keypoint_pixel_distance
        
        # Pre-extracted keypoints cache
        self.cached_keypoints = None
        self.cached_scores = None
        self.cached_gt_keypoints = None
        
    def extract_all_keypoints_once(self, 
                                   validation_heatmaps: List[torch.Tensor], 
                                   validation_gt_keypoints: List[List],
                                   min_threshold: float = 1e-6) -> None:
        """Extract keypoints once at very low threshold and cache results"""
        print(f"# Pre-extracting keypoints for {len(validation_heatmaps)} samples...")
        
        all_keypoints = []
        all_scores = []
        
        # Process in batches to save memory
        batch_size = 16
        for i in range(0, len(validation_heatmaps), batch_size):
            batch_end = min(i + batch_size, len(validation_heatmaps))
            batch_heatmaps = torch.cat(validation_heatmaps[i:batch_end], dim=0)
            
            # Extract with very low threshold to get all possible keypoints
            keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
                batch_heatmaps.float(),
                self.max_keypoints,
                self.minimal_keypoint_pixel_distance,
                abs_max_threshold=min_threshold,
                return_scores=True
            )
            
            all_keypoints.extend(keypoints)
            all_scores.extend(scores)
        
        self.cached_keypoints = all_keypoints
        self.cached_scores = all_scores
        self.cached_gt_keypoints = validation_gt_keypoints
        print(f"# Keypoint extraction completed")
    
    def evaluate_threshold_vectorized(self, threshold: float) -> float:
        """Fast threshold evaluation using pre-extracted keypoints"""
        if self.cached_keypoints is None:
            raise ValueError("Must call extract_all_keypoints_once first")
        
        # Create metrics
        temp_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) 
            for _ in range(len(self.cached_keypoints[0]))  # num channels
        ]
        
        # Process all samples
        for sample_idx in range(len(self.cached_keypoints)):
            sample_keypoints = self.cached_keypoints[sample_idx]
            sample_scores = self.cached_scores[sample_idx]
            sample_gt_keypoints = self.cached_gt_keypoints[sample_idx]
            
            # Filter keypoints by threshold for each channel
            for channel_idx in range(len(sample_keypoints)):
                # Fast threshold filtering
                if (len(sample_keypoints[channel_idx]) > 0 and 
                    len(sample_scores[channel_idx]) > 0):
                    
                    # Vectorized threshold filtering
                    scores_array = np.array(sample_scores[channel_idx])
                    valid_mask = scores_array > threshold
                    
                    detected_keypoints_channel = [
                        DetectedKeypoint(int(kp[0]), int(kp[1]), float(score))
                        for kp, score, valid in zip(
                            sample_keypoints[channel_idx], 
                            sample_scores[channel_idx], 
                            valid_mask
                        ) if valid
                    ]
                else:
                    detected_keypoints_channel = []
                
                # Ground truth keypoints for this channel
                gt_keypoints_channel = []
                if (channel_idx < len(sample_gt_keypoints) and 
                    len(sample_gt_keypoints[channel_idx]) > 0):
                    gt_keypoints_channel = [
                        Keypoint(int(k[0]), int(k[1])) 
                        for k in sample_gt_keypoints[channel_idx]
                    ]
                
                temp_metrics[channel_idx].update(detected_keypoints_channel, gt_keypoints_channel)
        
        # Compute mean AP
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
    
    def grid_search_optimal_threshold(self, 
                                      low: float = 1e-4, 
                                      high: float = 0.9, 
                                      num_coarse_steps: int = 20,
                                      num_fine_steps: int = 10) -> Tuple[float, float]:
        """Two-stage grid search: coarse then fine"""
        
        print(f"# Stage 1: Coarse grid search ({num_coarse_steps} points)")
        
        # Stage 1: Coarse grid search
        coarse_thresholds = np.linspace(low, high, num_coarse_steps)
        coarse_maps = []
        
        for threshold in coarse_thresholds:
            map_score = self.evaluate_threshold_vectorized(threshold)
            coarse_maps.append(map_score)
        
        # Find best coarse threshold
        best_coarse_idx = np.argmax(coarse_maps)
        best_coarse_threshold = coarse_thresholds[best_coarse_idx]
        best_coarse_map = coarse_maps[best_coarse_idx]
        
        print(f"# Best coarse threshold: {best_coarse_threshold:.6f} (mAP: {best_coarse_map:.4f})")
        
        # Stage 2: Fine search around best coarse result
        if best_coarse_idx == 0:
            fine_low = low
            fine_high = coarse_thresholds[1]
        elif best_coarse_idx == len(coarse_thresholds) - 1:
            fine_low = coarse_thresholds[-2]
            fine_high = high
        else:
            fine_low = coarse_thresholds[best_coarse_idx - 1]
            fine_high = coarse_thresholds[best_coarse_idx + 1]
        
        print(f"# Stage 2: Fine grid search in [{fine_low:.6f}, {fine_high:.6f}] ({num_fine_steps} points)")
        
        fine_thresholds = np.linspace(fine_low, fine_high, num_fine_steps)
        fine_maps = []
        
        for threshold in fine_thresholds:
            map_score = self.evaluate_threshold_vectorized(threshold)
            fine_maps.append(map_score)
        
        # Find best fine threshold
        best_fine_idx = np.argmax(fine_maps)
        best_threshold = fine_thresholds[best_fine_idx]
        best_map = fine_maps[best_fine_idx]
        
        print(f"# Optimal threshold: {best_threshold:.6f} (mAP: {best_map:.4f})")
        
        return best_threshold, best_map
    
    def parallel_threshold_evaluation(self, 
                                      thresholds: List[float]) -> List[float]:
        """Evaluate multiple thresholds in parallel (if needed)"""
        # This could be extended with multiprocessing if needed
        maps = []
        for threshold in thresholds:
            map_score = self.evaluate_threshold_vectorized(threshold)
            maps.append(map_score)
        return maps


# Integration into KeypointDetector
def optimized_threshold_search_method(self, 
                                      validation_heatmaps: List[torch.Tensor], 
                                      validation_gt_keypoints: List[List],
                                      low: float = None, 
                                      high: float = None) -> Tuple[float, float]:
    """Optimized replacement for ternary_search_optimal_threshold"""
    
    # Use instance parameters if not provided
    if low is None:
        low = self.threshold_search_low
    if high is None:
        high = self.threshold_search_high
    
    # Create optimizer
    optimizer = OptimizedThresholdSearch(
        self.maximal_gt_keypoint_pixel_distances,
        self.max_keypoints,
        self.minimal_keypoint_pixel_distance
    )
    
    # Pre-extract keypoints once
    optimizer.extract_all_keypoints_once(validation_heatmaps, validation_gt_keypoints)
    
    # Find optimal threshold using grid search
    optimal_threshold, best_map = optimizer.grid_search_optimal_threshold(low, high)
    
    return optimal_threshold, best_map


# Memory-efficient variant for very large datasets
class MemoryEfficientThresholdSearch(OptimizedThresholdSearch):
    """Memory-efficient version that doesn't cache all keypoints"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_heatmaps = None
        self.validation_gt_keypoints = None
    
    def set_validation_data(self, heatmaps: List[torch.Tensor], gt_keypoints: List[List]):
        """Store references to validation data instead of caching keypoints"""
        self.validation_heatmaps = heatmaps
        self.validation_gt_keypoints = gt_keypoints
    
    def evaluate_threshold_on_demand(self, threshold: float) -> float:
        """Extract keypoints on-demand for each threshold (still faster than original)"""
        if self.validation_heatmaps is None:
            raise ValueError("Must call set_validation_data first")
        
        temp_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) 
            for _ in range(len(self.validation_heatmaps[0][0]))  # num channels
        ]
        
        # Process in batches to save memory
        batch_size = 8
        for i in range(0, len(self.validation_heatmaps), batch_size):
            batch_end = min(i + batch_size, len(self.validation_heatmaps))
            batch_heatmaps = torch.cat(self.validation_heatmaps[i:batch_end], dim=0)
            
            keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
                batch_heatmaps.float(),
                self.max_keypoints,
                self.minimal_keypoint_pixel_distance,
                abs_max_threshold=threshold,
                return_scores=True
            )
            
            # Update metrics for this batch
            for batch_idx, (sample_keypoints, sample_scores) in enumerate(zip(keypoints, scores)):
                sample_gt_keypoints = self.validation_gt_keypoints[i + batch_idx]
                
                for channel_idx in range(len(sample_keypoints)):
                    detected_keypoints_channel = []
                    if (len(sample_keypoints[channel_idx]) > 0 and 
                        len(sample_scores[channel_idx]) > 0):
                        detected_keypoints_channel = [
                            DetectedKeypoint(int(kp[0]), int(kp[1]), float(score))
                            for kp, score in zip(sample_keypoints[channel_idx], sample_scores[channel_idx])
                        ]
                    
                    gt_keypoints_channel = []
                    if (channel_idx < len(sample_gt_keypoints) and 
                        len(sample_gt_keypoints[channel_idx]) > 0):
                        gt_keypoints_channel = [
                            Keypoint(int(k[0]), int(k[1])) 
                            for k in sample_gt_keypoints[channel_idx]
                        ]
                    
                    temp_metrics[channel_idx].update(detected_keypoints_channel, gt_keypoints_channel)
        
        # Compute mean AP
        total_ap = 0.0
        total_count = 0
        
        for channel_metrics in temp_metrics:
            channel_aps = channel_metrics.compute()
            if len(channel_aps) > 0:
                total_ap += sum(channel_aps.values())
                total_count += len(channel_aps)
            channel_metrics.reset()
        
        return total_ap / max(total_count, 1)