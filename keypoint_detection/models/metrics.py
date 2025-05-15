"""
Implementation of (mean) Average Precision metric for 2D keypoint detection.
"""

from __future__ import annotations  # allow typing of own class objects

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
from torchmetrics import Metric
from torch import nn

@dataclass
class Keypoint:
    """A simple class datastructure for Keypoints,
    dataclass is chosen over named tuple because this class is inherited by other classes
    """

    u: int
    v: int

    def l2_distance(self, keypoint: Keypoint):
        return math.sqrt((self.u - keypoint.u) ** 2 + (self.v - keypoint.v) ** 2)


@dataclass
class DetectedKeypoint(Keypoint):
    probability: float


@dataclass(unsafe_hash=True)
class ClassifiedKeypoint(DetectedKeypoint):
    """
    DataClass for a classified keypoint, where classified means determining if the detection is a True Positive of False positive,
     with the given treshold distance and the gt keypoints from the frame

    a hash is required for torch metric
    cf https://github.com/PyTorchLightning/metrics/blob/2c8e46f87cb67186bff2c7b94bf1ec37486873d4/torchmetrics/metric.py#L570
    unsafe_hash -> dirty fix to allow for hash w/o explictly telling python the object is immutable.
    """

    threshold_distance: int
    true_positive: bool


def keypoint_classification(
    detected_keypoints: List[DetectedKeypoint],
    ground_truth_keypoints: List[Keypoint],
    threshold_distance: int,
) -> List[ClassifiedKeypoint]:
    """Classifies keypoints of a **single** frame in True Positives or False Positives by searching for unused gt keypoints in prediction probability order
    that are within distance d of the detected keypoint (greedy matching).

    Args:
        detected_keypoints (List[DetectedKeypoint]): The detected keypoints in the frame
        ground_truth_keypoints (List[Keypoint]): The ground truth keypoints of a frame
        threshold_distance: maximal distance in pixel coordinate space between detected keypoint and ground truth keypoint to be considered a TP

    Returns:
        List[ClassifiedKeypoint]: Keypoints with TP label.
    """
    classified_keypoints: List[ClassifiedKeypoint] = []

    ground_truth_keypoints = copy.deepcopy(
        ground_truth_keypoints
    )  # make deep copy to do local removals (pass-by-reference..)

    for detected_keypoint in sorted(detected_keypoints, key=lambda x: x.probability, reverse=True):
        matched = False
        for gt_keypoint in ground_truth_keypoints:
            distance = detected_keypoint.l2_distance(gt_keypoint)
            # add small epsilon to avoid numerical errors
            if distance <= threshold_distance + 1e-5:
                classified_keypoint = ClassifiedKeypoint(
                    detected_keypoint.u,
                    detected_keypoint.v,
                    detected_keypoint.probability,
                    threshold_distance,
                    True,
                )
                matched = True
                # remove keypoint from gt to avoid muliple matching
                ground_truth_keypoints.remove(gt_keypoint)
                break
        if not matched:
            classified_keypoint = ClassifiedKeypoint(
                detected_keypoint.u,
                detected_keypoint.v,
                detected_keypoint.probability,
                threshold_distance,
                False,
            )
        classified_keypoints.append(classified_keypoint)

    return classified_keypoints


def calculate_precision_recall(
    classified_keypoints: List[ClassifiedKeypoint], total_ground_truth_keypoints: int
) -> Tuple[List[float], List[float]]:
    """Calculates precision recall points on the curve for the given keypoints by varying the treshold probability to all detected keypoints
     (i.e. by always taking one additional keypoint als a predicted event)

    Note that this function is tailored towards a Detector, not a Classifier. For classifiers, the outputs contain both TP, FP and FN. Whereas for a Detector the
    outputs only define the TP and the FP; the FN are not contained in the output as the point is exactly that the detector did not detect this event.

    A detector is a ROI finder + classifier and the ROI finder could miss certain regions, which results in FNs that are hence never passed to the classifier.

    This also explains why the scikit average_precision function states it is for Classification tasks only. Since it takes "total_gt_events" to be the # of positive_class labels.
    The function can however be used by using as label (TP = 1, FP = 0) and by then multiplying the result with TP/(TP + FN) since the recall values are then corrected
    to take the unseen events (FN's) into account as well. They do not matter for precision calcultations.
    Args:
        classified_keypoints (List[ClassifiedKeypoint]):
        total_ground_truth_keypoints (int):

    Returns:
        Tuple[List[float], List[float]]: precision, recall entries. First entry is (1,0); last entry is (0,1).
    """
    precision = [1.0]
    recall = [0.0]

    true_positives = 0
    false_positives = 0

    for keypoint in sorted(classified_keypoints, key=lambda x: x.probability, reverse=True):
        if keypoint.true_positive:
            true_positives += 1
        else:
            false_positives += 1

        precision.append(_zero_aware_division(true_positives, (true_positives + false_positives)))
        recall.append(_zero_aware_division(true_positives, total_ground_truth_keypoints))

    precision.append(0.0)
    recall.append(1.0)
    return precision, recall


def calculate_ap_from_pr(precision: List[float], recall: List[float]) -> float:
    """Calculates the Average Precision using the AUC definition (COCO-style)

    # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    # AUC AP.

    Args:
        precision (List[float]):
        recall (List[float]):

    Returns:
        (float): average precision (between 0 and 1)
    """

    smoothened_precision = copy.deepcopy(precision)

    for i in range(len(smoothened_precision) - 2, 0, -1):
        smoothened_precision[i] = max(smoothened_precision[i], smoothened_precision[i + 1])

    ap = 0
    for i in range(len(recall) - 1):
        ap += (recall[i + 1] - recall[i]) * smoothened_precision[i + 1]

    return ap


class KeypointAPMetric(Metric):
    full_state_update = True  # Important for distributed training

    def __init__(self, keypoint_threshold_distance: float, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.keypoint_threshold_distance = keypoint_threshold_distance

        # Initialize tensor states with device=None (they'll be moved to the correct device)
        self.add_state("tp_count", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("fp_count", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_ground_truth_keypoints", default=torch.tensor(0.), dist_reduce_fx="sum")
        
        # For probability scores, initialize empty tensors
        self.add_state("all_probabilities", default=[], dist_reduce_fx=None)
        self.add_state("all_is_tp", default=[], dist_reduce_fx=None)
        
        # For local tracking only - will be processed before compute()
        self._classified_keypoints = []

    def update(self, detected_keypoints: List[DetectedKeypoint], gt_keypoints: List[Keypoint]):
        # Classify each keypoint as TP or FP
        classified_img_keypoints = keypoint_classification(
            detected_keypoints, gt_keypoints, self.keypoint_threshold_distance
        )
        
        # Store locally for processing later
        self._classified_keypoints.extend(classified_img_keypoints)
        
        # Update tensor states
        for keypoint in classified_img_keypoints:
            # Track probability and whether it's a TP
            self.all_probabilities.append(keypoint.probability)
            self.all_is_tp.append(1.0 if keypoint.true_positive else 0.0)
            
            if keypoint.true_positive:
                self.tp_count += 1
            else:
                self.fp_count += 1
                
        self.total_ground_truth_keypoints += len(gt_keypoints)

    def compute(self):
        # Convert lists to tensors for synchronization, ensuring they're on the right device
        if self.all_probabilities and self.all_is_tp:
            all_probabilities = torch.tensor(self.all_probabilities, device=self.tp_count.device)
            all_is_tp = torch.tensor(self.all_is_tp, device=self.tp_count.device)
        else:
            # Create empty tensors with the correct dimensions and device
            all_probabilities = torch.tensor([], device=self.tp_count.device)
            all_is_tp = torch.tensor([], device=self.tp_count.device)
            
        # These will be properly synced across devices
        tp_count = self.tp_count
        fp_count = self.fp_count
        total_gt_count = self.total_ground_truth_keypoints
        
        # If no detections or ground truth, return 0
        if total_gt_count == 0 or (tp_count == 0 and fp_count == 0):
            return 0.0
            
        # Compute AP using synced tensors (all now on the proper device)
        if len(all_probabilities) == 0:
            return 0.0
            
        # Sort by probability
        sorted_indices = torch.argsort(all_probabilities, descending=True)
        sorted_is_tp = all_is_tp[sorted_indices]
        
        # Calculate precision and recall points
        precision = [1.0]
        recall = [0.0]
        
        tp_sum = 0
        fp_sum = 0
        
        for is_tp in sorted_is_tp:
            if is_tp > 0.5:  # TP
                tp_sum += 1
            else:  # FP
                fp_sum += 1
                
            precision.append(tp_sum / (tp_sum + fp_sum) if tp_sum + fp_sum > 0 else 0)
            recall.append(tp_sum / total_gt_count.item() if total_gt_count.item() > 0 else 0)
            
        precision.append(0.0)
        recall.append(1.0)
        
        # Calculate AP using your existing function
        ap = calculate_ap_from_pr(precision, recall)
        return ap
class KeypointAPMetrics(Metric):
    full_state_update = True  # Important for distributed training
    
    def __init__(self, keypoint_threshold_distances: List[int], dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Register metrics as a ModuleList so PyTorch handles them properly
        self.ap_metrics = nn.ModuleList([
            KeypointAPMetric(dst, dist_sync_on_step) 
            for dst in keypoint_threshold_distances
        ])
        
        self.keypoint_threshold_distances = keypoint_threshold_distances

    def update(self, detected_keypoints: List[DetectedKeypoint], gt_keypoints: List[Keypoint]):
        for metric in self.ap_metrics:
            metric.update(detected_keypoints, gt_keypoints)

    def compute(self) -> Dict[float, float]:
        result_dict = {}
        for i, metric in enumerate(self.ap_metrics):
            result_dict.update({float(self.keypoint_threshold_distances[i]): metric.compute()})
        return result_dict

    def reset(self) -> None:
        for metric in self.ap_metrics:
            metric.reset()


def _zero_aware_division(num: float, denom: float) -> float:
    if num == 0:
        return 0
    if denom == 0 and num != 0:
        return float("inf")
    else:
        return num / denom


# if __name__ == "__main__":
#     print(
#         check_forward_full_state_property(
#             KeypointAPMetric,
#             init_args={"keypoint_threshold_distance": 2.0},
#             input_args={"detected_keypoints": [DetectedKeypoint(10, 20, 0.02)], "gt_keypoints": [Keypoint(10, 23)]},
#         )
#     )


# if __name__ == "__main__":
#     import numpy as np
#     from sklearn.metrics import average_precision_score, precision_recall_curve
#     import matplotlib.pyplot as plt

#     y_true = np.array([1, 1, 0, 1,0,0,0,0])
#     y_scores = np.array([0.1, 0.4, 0.35, 0.8,0.01,0.01,0.01,0.01])

#     y_true = np.random.randint(0,2,100)
#     y_scores = np.random.rand(100)
#     sklearn_precisions, sklearn_recalls, _ = precision_recall_curve(y_true, y_scores)
#     sklearnAP = average_precision_score(y_true, y_scores)

#     print(f"sklearn AP: {sklearnAP}")
#     my_precisions, my_recalls  = calculate_precision_recall([ClassifiedKeypoint(None,None,y_scores[i],None,y_true[i]) for i in range(len(y_true))], sum(y_true))
#     myAP = calculate_ap_from_pr(my_precisions, my_recalls)
#     print(f"my AP: {myAP}")

#     plt.plot(sklearn_recalls, sklearn_precisions, label=f"sklearn AP: {sklearnAP}")
#     plt.plot(my_recalls, my_precisions, label=f"my AP: {myAP}")
#     plt.legend()
#     plt.savefig("test.png")
