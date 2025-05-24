#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from typing import List, Union

from src import postprocess
from src.utils import convert


#####################################
# Functions
#####################################

# ----------------------------
# Intersection Over Union
# ----------------------------
def bbox_area(bboxes: torch.Tensor) -> torch.Tensor:
    '''
    Computes the area of bounding boxes.
    
    Args:
        bboxes (torch.Tensor): Tensor of shape (..., num_bboxes, 4+),
                               where the last dimension represents bounding boxes in the format:
                               (x_min, y_min, x_max, y_max).

    Returns:
        torch.Tensor: Tensor of shape (..., num_bboxes) containing
                      the area of each bounding box.
    '''
    
    # Clamp assigns an area of 0 to malformed bboxes
    bbox_widths = (bboxes[..., 2] - bboxes[..., 0]).clamp(min = 0) # Width is x_max - x_min
    bbox_heights = (bboxes[..., 3] - bboxes[..., 1]).clamp(min = 0) # Height is y_max - y_min
    return bbox_widths * bbox_heights

def calc_ious(pred_bboxes: torch.Tensor, targ_bboxes: torch.Tensor) -> torch.Tensor:
    '''
    Computes the Intersection over Union (IoU) between predicted and ground-truth bounding boxes
    
    Args:
        pred_bboxes (torch.Tensor): Tensor of shape (..., num_pred, 4+)
                                    containing predicted bounding boxes in 
                                    (x_min, y_min, x_max, y_max) format as the first 4 elements.
        targ_bboxes (torch.Tensor): Tensor of shape (..., num_gt, 4+)
                                    containing ground-truth bounding boxes in 
                                    (x_min, y_min, x_max, y_max) format as the first 4 elements.

    Returns:
        torch.Tensor: IoU values of shape (..., num_pred, num_gt), 
                      where the entry at [i, j] gives 
                      the IoU between the i-th predicted box and
                      j-th ground-truth box.
    '''
    
    # Upper-left and bottom-right corners of intersections
    inter_ul = torch.max(targ_bboxes[..., None, :, :2], pred_bboxes[..., :, None, :2])
    inter_br = torch.min(targ_bboxes[..., None, :, 2:4], pred_bboxes[..., :, None, 2:4])
    
    # Intersection width and height -> intersection area
    # Clamp to 0 avoid negative values and indicates no overlap
    inter_lengths = (inter_br - inter_ul).clamp(min = 0)
    inter_areas = inter_lengths[..., 0] * inter_lengths[..., 1]
    
    # Union areas: area(A) + area(B) - area(intersection)
    union_areas = (bbox_area(targ_bboxes)[..., None, :] + 
                   bbox_area(pred_bboxes)[..., :, None] - 
                   inter_areas)

    # Shape: (num_pred, num_gt)
    return inter_areas / (union_areas + 1e-7)


# ----------------------------
# Mean Average Precision
# ----------------------------
def calc_mAP(pred_dicts: List[dict], 
             targ_dicts: List[dict], 
             thresholds: List[float] = [0.5]):
    '''
    Computes the mean Average Precision (mAP) metric across IoU thresholds, 
    given a list of prediction and target dictionaries.
    
    Note: `torchmetrics.detection.mean_ap.MeanAveragePrecision` is faster.

    Args:
        pred_dicts (List[dict]): A list of length batch_size, containing prediction dictionaries for each image sample.
                                The keys of each prediction dictionary are named to be compatible with torchmetrics.detection.mean_ap.
                                They are as follows:
                                - boxes (torch.Tensor): The predicted bounding boxes in (x_min, y_min, x_max, y_max) format.
                                                        Shape is (num_filtered_bboxes, 4).
                                - labels (torch.Tensor): The predicted class labels for the bounding boxes in `bboxes`.
                                                            Shape is (num_filtered_bboxes,)
                                - scores (torch.Tensor): The class confidence scores for `labels`. 
                                                            This is defined as P(class_i) * IoU^{truth}_{pred}.
                                                            Shape is (num_filtered_bboxes,)
        targ_dicts (List[dict]): A list of length batch_size containing target dictionaries.
                                 The keys of each prediction dictionary are named to be compatible with torchmetrics.detection.mean_ap.
                                 They are as follows:
                                    - boxes (torch.Tensor): The bounding boxes in (x_min, y_min, x_max, y_max) format.
                                                            Shape is (num_objects, 4).
                                    - labels (torch.Tensor): The class labels for the bounding boxes in `bboxes`.
                                                             Shape is (num_objects,).
        thresholds (List[float]): A list of IoU thresholds used for mAP calculations. Default is [0.5].

    Returns:
        mAP_dict (dict): A dictionary containing the calculated mAP values.
                         The keys are:
                            - mAP (torch.Tensor): The overall mAP value calculated across all classes and IoU thresholds.
                            - mAP_50 (torch.Tensor): The overall mAP@50 value calculated across all classes at IoU threshold = 0.5.
                                      If 0.5 is not in thresholds, this is nan.
                            - mAP_75 (torch.Tensor): The overall mAP@75 value calculated across all classes at IoU threshold = 0.75.
                                      If 0.75 is not in thresholds, this is nan.
                            - label_APs (torch.Tensor): List of per-class AP values across all IoU thresholds.
                            - targ_labels (torch.Tensor): List of unique target class labels that correspond to label_APs
    mAP references: 
        - https://wiki.cloudfactory.com/docs/mp-wiki/metrics/map-mean-average-precision
    '''
    targ_labels, pred_labels, pred_bboxes, pred_scores, pred_img_idxs = [], [], [], [], []
    device = targ_dicts[0]['boxes'].device

    for i in range(len(targ_dicts)):
        targ_res, pred_res = targ_dicts[i], pred_dicts[i]

        targ_labels.append(targ_res['labels'])

        pred_labels.append(pred_res['labels'])
        pred_bboxes.append(pred_res['boxes'])
        pred_scores.append(pred_res['scores'])
        pred_img_idxs.append(torch.tensor(i, device = device).repeat(len(pred_res['labels'])))

    # Concatenate predictions across all images and sort by confidence scores
    pred_scores = torch.concat(pred_scores, dim = 0)
    pred_scores, order = torch.sort(pred_scores, descending = True)

    pred_labels = torch.cat(pred_labels, dim = 0)[order]
    pred_bboxes = torch.cat(pred_bboxes, dim = 0)[order]
    pred_img_idxs = torch.cat(pred_img_idxs, dim = 0)[order]

    # Get all unique target labels
    targ_labels = torch.cat(targ_labels, dim = 0)
    unique_targ_labels = torch.unique(targ_labels)
    
    # Lists to keep track of class-wise APs and common mAP@ values
    label_APs, mAP_50_APs, mAP_75_APs = [], [], []

    # Calculate AP for all unqiue target class labels
    for label in unique_targ_labels:
        label_cache = {}

        pred_mask = pred_labels.eq(label)  # Mask to filter for only predicted bboxes with current class label
        num_preds = pred_mask.sum() # Number of predictions after filtering
        num_targs = targ_labels.eq(label).sum() # Number of positive cases (TP + FN)
        
        # Calculate AP over all thresholds for the class label
        threshold_APs = []
        for threshold in thresholds:
            for targ_res in targ_dicts:
                # Reset matched for the threshold
                targ_res['matched'] = torch.zeros(len(targ_res['labels']),  dtype = bool, device = device)
            
            tp = torch.zeros(num_preds, device = device)
            fp = torch.zeros(num_preds, device = device)
            
            # Loop through predicted bboxes and match them to target bboxes
            for i, (img_idx, pred_bbox) in enumerate(zip(pred_img_idxs[pred_mask], pred_bboxes[pred_mask])):
                
                targ_res = targ_dicts[img_idx] # Get target bboxes for the relevant image
                targ_idxs = label_cache.setdefault(
                    f'targ_{img_idx}',
                    torch.where(targ_res['labels'].eq(label))[0]
                )
                targ_bboxes = targ_res['boxes'][targ_idxs] # Filter for only target bboxes with current class label
                
                # No target bboxes in the relevant image => False Positive
                if len(targ_bboxes) == 0:
                    fp[i] = 1

                else:
                    ious = calc_ious(pred_bbox.unsqueeze(0), targ_bboxes)
                    max_iou, max_idx = ious.max(dim = -1)
                    
                    # Max IoU exceeds threshold and corresponding target bbox hasn't been matched yet => True Positive
                    if (max_iou >= threshold) and (not targ_res['matched'][targ_idxs[max_idx]]):
                        tp[i] = 1
                        targ_res['matched'][targ_idxs[max_idx]] = True
                    
                    # Either max IoU doesn't exceed threshold or the target bbox has already been matched
                        # => False Positive
                    else:
                        fp[i] = 1

            tp = tp.cumsum(0).float()
            fp = fp.cumsum(0).float()

            # Pad left side of recall with 0
            recall = tp / num_targs # Recall = TP / (num_positive_cases)
            recall = torch.concat([torch.tensor([0.0], device = device), recall])

            precision = tp / (tp + fp) # Precision = TP / (TP + FP)
            precision = torch.concat([precision[0:1], precision]).to(device)

            # Smooth out precisions, ensuring it is monotonically decreasing
            for i in range(len(precision) - 2, -1, -1):
                precision[i] = torch.max(precision[i], precision[i + 1])
            
            # Use trapezoidal rule to calculate AUC of precision-recall curve
            ap = torch.trapz(precision, recall).item()
            threshold_APs.append(ap)
            
            if threshold == 0.5:
                mAP_50_APs.append(ap)
            elif threshold == 0.75:
                mAP_75_APs.append(ap)
                
        # Average AP across thresholds
        label_APs.append(sum(threshold_APs) / len(threshold_APs))
    
    mAP_dict = {
        'mAP': torch.tensor(label_APs).mean(),
        'mAP_50': torch.tensor(mAP_50_APs).mean() if mAP_50_APs else torch.nan,
        'mAP_75': torch.tensor(mAP_75_APs).mean() if mAP_75_APs else torch.nan,
        'label_APs': torch.tensor(label_APs),
        'targ_labels': unique_targ_labels
    }
        
    return mAP_dict

def calc_dataset_mAP(model: nn.Module, 
                     dataloader: DataLoader, 
                     obj_threshold: float = 0.25,
                     nms_threshold: float = 0.5,
                     mAP_thresholds: List[float] = [0.5],
                     device: Union[torch.device, str] = 'cpu'):
    '''
    Evaluates a YOLOv1 model on a given dataset using the MeanAveragePrecision metric
    from `torchmetrics.detection`. 
    This includes the metrics: mean Average Precision (mAP) and mean Average Recall (mAR).

    Args:
        model (nn.Module): An instance of a YOLOv1 model to bounding boxes and class labels.
                           This model should already be moved to the specified `device`.
        dataloader (data.Dataloader): The dataloader used to transform and load the dataset in batches.
        obj_threshold (float): Threshold to filter out low predicted object confidence scores. Default is 0.25.
        nms_threshold (float): The IoU threshold used when performing Non-Maximum Suppression (NMS). Default is 0.5.
        mAP_thresholds (List[float]): A list of IoU thresholds used for mAP calculations. Default is [0.5].
        device (torch.device or str): The device to perform calculations on. Default is 'cpu'.
    
    Returns:
        dict: Dictionary containing mAP and mean Average Recall (mAR) metrics as computed by
              `torchmetrics.detection.mean_ap.MeanAveragePrecision`.

              The most important keys are:
                - map (torch.Tensor): The overall mAP value calculated across all classes and IoU thresholds.
                - map_per_class (torch.Tensor): List of per-class AP values across all IoU thresholds.

              For more information, see:
                https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
    '''
    mAP = MeanAveragePrecision(box_format = 'xyxy', 
                               class_metrics = True, 
                               iou_thresholds = mAP_thresholds)
    model.eval()
    for imgs, targs in dataloader:
        imgs, targs = imgs.to(device), targs.to(device)

        targ_dicts = postprocess.decode_targets_yolov1(targs, S = model.S, B = model.B)

        with torch.inference_mode():
            pred_dicts = predict_yolov1(model, imgs, obj_threshold, nms_threshold)

        mAP.update(pred_dicts, targ_dicts)

    return mAP.compute()


# ----------------------------
# Prediction
# ----------------------------
def predict_yolov1(
        model: nn.Module, 
        X: torch.Tensor, 
        obj_threshold: float = 0.25,
        nms_threshold: float = 0.5
    ) -> List[dict]:
    '''
    Uses a YOLOv1 model to predicted the bounding boxes and class labels for a batch of preprocessed images.
    The predictions are first filtered by object confidence score and then filtered by Non-Maximum Suppression (NMS).
    The bounding box predictions are returned in (x_min, y_min, x_max, y_max) format.

    Args:
        model (nn.Module): An instance of a YOLOv1 model to predict bounding boxes and class labels.
        X (torch.Tensor): A tensor containing the batch of preprocessed images to predict on.
                          This should be on the same device as the model.
                          Shape is (batch_size, channels, height, width). 
                          For a standard YOLOv1 model, this has shape (batch_size, 3, 448, 448).
        obj_threshold (float): Threshold to filter out low predicted object confidence scores. Default is 0.25.
        nms_threshold (float): The IoU threshold used when performing NMS. Default is 0.5.

    Returns:
        pred_dicts (List[dict]): A list of length batch_size, containing prediction dictionaries for each image sample in X.
                                 The keys of each prediction dictionary are named to be compatible with torchmetrics.detection.mean_ap.
                                 They are as follows:
                                    - boxes (torch.Tensor): The predicted bounding boxes in (x_min, y_min, x_max, y_max) format.
                                                            Shape is (num_filtered_bboxes, 4).
                                    - labels (torch.Tensor): The predicted class labels for the bounding boxes in `bboxes`.
                                                             Shape is (num_filtered_bboxes,)
                                    - scores (torch.Tensor): The class confidence scores for `labels`. 
                                                             This is defined as P(class_i) * IoU^{truth}_{pred}.
                                                             Shape is (num_filtered_bboxes,)
    '''
    assert len(X.shape) == 4, (
        'Incorrect number of dimensions for X. Expecting shape (batch_size, channels, height, width).'
    )

    device = X.device
    S, B = model.S, model.B
    
    model.eval()
    with torch.inference_mode():
        preds = model(X) # Shape: (batch_size, S, S, B*5 + C)

    grid_i, grid_j = torch.meshgrid(torch.arange(S), torch.arange(S), indexing = 'ij')
    grid_i = grid_i[None, ..., None].to(device)  # Shape: (1, S, S, 1)
    grid_j = grid_j[None, ..., None].to(device)  # Shape: (1, S, S, 1)

    pred_dicts = []
    # Loop over the batch to perform confidence score filtering and NMS
    for i in range(preds.shape[0]):
        pred_res = {}
        pred_samp = preds[i].unsqueeze(0) # Shape: (1, S, S, B*5 + C)

        # pred_bboxes shape: (1, S, S, B, 5)
        # pred_prob_dist shape: (1, S, S, B, C)
        pred_bboxes, pred_prob_dist = postprocess.decode_logits_yolov1(pred_samp, S, B, split_output = True)

        pred_bboxes = convert.yolov1_to_corner_format(pred_bboxes, grid_i, grid_j, S)

        # Filter out low predicted confidence scores
        conf_mask = pred_bboxes[..., -1] >= obj_threshold
        pred_bboxes = pred_bboxes[conf_mask] # Shape: (num_bboxes, 5)
        pred_probs, pred_labels = pred_prob_dist[conf_mask].max(dim = -1) # Shape: (num_bboxes,), (num_bboxes,)

        # Class confidence scores, 
            # defined as P(class_i) * IoU^{truth}_{pred} = P(class_i|object) * object confidence score
        pred_scores = pred_probs * pred_bboxes[:, 4]

        keep_idxs = batched_nms(boxes = pred_bboxes[:, :4], 
                                scores = pred_scores, 
                                idxs = pred_labels, 
                                iou_threshold = nms_threshold)

        # Filter bboxes, labels, and confidence scores
        pred_res['boxes'] = pred_bboxes[:, :4][keep_idxs] # Shape (num_filtered_bboxes, 4)
        pred_res['labels'] = pred_labels[keep_idxs] # Shape (num_filtered_bboxes,)
        pred_res['scores'] = pred_scores[keep_idxs] # Shape (num_filtered_bboxes,)

        pred_dicts.append(pred_res)

    return pred_dicts
