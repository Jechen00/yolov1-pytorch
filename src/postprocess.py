#####################################
# Imports & Dependencies
#####################################
import torch
import torch.nn.functional as F

from typing import Union, Tuple, List

from src import evaluate
from src.utils import convert


#####################################
# Functions
#####################################
def decode_logits_yolov1(
        pred_logits: torch.Tensor, 
        S: int = 7, 
        B: int = 2, 
        split_output: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''
    Decodes/postprocesses raw YOLOv1 model outputs (logits) as follows:
        - Applies sigmoid to bbox predictions + confidence.
        - Applies softmax to class logits.
    
    Optionally splits the output into bounding box predictions and class probabilities.

    Args:
        pred_logits (torch.Tensor): Tensor of logits from YOLOv1 output. 
                                    Shape is (batch_size, S, S, B*5 + C), where C is the number of classes.
        S (int): Grid size.
        B (int): Number of bounding boxes per grid cell.
        split_output (bool): Whether to return separate tensors for bboxes and class predictions.

    Returns:
        If split_output is False:
            - Full postprocessed tensor of shape (batch_size, S, S, B*5 + C),
        Else:
            - The tuple (bbox_preds, class_probs) with shapes:
                - bbox_preds: (batch_size, S, S, B, 5)
                - class_probs: (batch_size, S, S, B, C)
    '''
    bbox_logits = pred_logits[..., :B*5]
    class_logits = pred_logits[..., B*5:]

    bbox_preds = torch.sigmoid(bbox_logits) # Bbox predictions and confidence should be within [0, 1]
    class_probs = F.softmax(class_logits, dim = -1) # Class predictions need to be softmaxed to be a proper distribution

    if not split_output:
        return torch.cat([bbox_preds, class_probs], dim = -1)

    else:
        bbox_preds = bbox_preds.view(-1, S, S, B, 5)
        class_probs = class_probs.unsqueeze(3).repeat(1, 1, 1, B, 1)
        return bbox_preds, class_probs
    
def decode_targets_yolov1(targs: torch.Tensor, S: int = 7, B: int = 2) -> List[dict]:
    '''
    Decodes a batch of YOLOv1 targets into a list of dictionaries containing bounding boxes and labels.
    
    Args:
        targs (Tensor): Tensor of shape (batch_size, S, S, B*5 + C)
    
    Returns:
        List[dict]: A list of length batch_size containing target dictionaries.
                    The keys of each prediction dictionary are named to be compatible with torchmetrics.detection.mean_ap.
                    They are as follows:
                        - boxes (torch.Tensor): The bounding boxes in (x_min, y_min, x_max, y_max) format.
                                                Shape is (num_objects, 4).
                        - labels (torch.Tensor): The class labels for the bounding boxes in `bboxes`.
                                                 Shape is (num_objects,).
    '''
    
    assert len(targs.shape) == 4, (
        'Incorrect number of dimensions for targs. Expecting shape (batch_size, S, S, B*5 + C).'
    )
    
    device = targs.device
    
    grid_i, grid_j = torch.meshgrid(torch.arange(S), torch.arange(S), indexing = 'ij')
    grid_i = grid_i[None, ..., None].to(device)  # Shape: (1, S, S, 1)
    grid_j = grid_j[None, ..., None].to(device)  # Shape: (1, S, S, 1)
        
    targ_dicts = []
    for i in range(targs.shape[0]):
        targ_res = {}
        targ_sample = targs[i].unsqueeze(0)
        
        # Shape: (1, S, S)
        obj_mask = targ_sample[..., 4].bool() # Mask indicating which grid cells have objects
        
        bboxes = targ_sample[..., :B*5].clone().view(1, S, S, B, 5) # Shape: (1, S, S, B, 5)
        bboxes = convert.yolov1_to_corner_format(bboxes, grid_i, grid_j, S)

        # Get bboxes in object cells
        # Note: For targets, only the first bbox has meaningful non-zero values for object cells
        bboxes = bboxes[obj_mask][:, 0]
        targ_res['boxes'] = bboxes[:, :4] # Shape: (num_bboxes, 4)

        # Get labels in object cells
        targ_probs = targ_sample[..., B*5:]
        
        # Decode labels (argmax over one-hot class vector)
        targ_res['labels'] = targ_probs[obj_mask].argmax(dim = -1) # Shape: (num_bboxes,)

        targ_dicts.append(targ_res)
        
    return targ_dicts

def non_max_suppression(bboxes: torch.Tensor, 
                        labels: torch.Tensor, 
                        scores: torch.Tensor,
                        threshold: float = 0.5) -> list[int]:   
    '''
    Performs class-wise Non-Maximum Suppression (NMS).
    Note: batched_nms from torchvision.ops is faster.
    Reference: https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
    
    Args:
        bboxes (torch.Tensor): A tensor of bounding boxes to perform NMS on. 
                               The shape should be (num_bboxes, 4), 
                               where the last dimension is in the format (x_min, y_min, x_max, y_max).
        labels (torch.Tensor): A tensor of class labels for the corresponsing bounding boxes in `bboxes`. 
                               Shape is (num_bboxes,).
        scores (torch.Tensor): A tensor of class confidence scores for the corresponsing bounding boxes in `bboxes`.
                               shape is (num_bboxes,).
        threshold (float): The IoU hreshold, over which two bounding boxes are considered too similar. 
                           Default is 0.5.

    Returns:
        keep_idxs (list): A list of indices indicating which bounding boxes in `bboxes` 
                          to keep after NMS.
    '''
    keep_idxs = []
    
    for cls in torch.unique(labels):
        cls_idxs = torch.where(labels.eq(cls))[0] # Indices where bbox_labels are equal to cls
        cls_bboxes = bboxes[cls_idxs] # All bboxes that predict the class; Shape: (num_matched, 5)

        # Indices to sort confidence scores in descending order
        sort_idxs = torch.argsort(scores[cls_idxs], descending = True)

        while len(sort_idxs) > 0:
            # Add the index with max confidence score to the list of indices we keep
            keep_idxs.append(cls_idxs[sort_idxs[0]].item())

            if len(sort_idxs) == 1: break

            # cls_bboxes[sort_idxs[0:1]] -> bbox with max confidence score
            # cls_bboxes[sort_idxs[1:]] -> The rest of the bboxes
            ious = evaluate.calc_ious(cls_bboxes[sort_idxs[0:1]], cls_bboxes[sort_idxs[1:]])

            # Filter out indices where IoU exceeds nms_threshold
            sort_idxs = sort_idxs[1:][ious.squeeze(0) <= threshold]
            
    return keep_idxs