#####################################
# Functions
#####################################
import torch


#####################################
# Functions
#####################################
def corner_to_center_format(bboxes):
    '''
    Converts bbox coordinates from corner (x_min, y_min, x_max, y_max) format
    to center (x_center, y_center, width, height) format.

    In PyTorch BoundingBoxes convention: XYXY -> CXCYWH

    Args:
        bboxes: Bbox coordinates in corner format.
                Shape is (..., 4+), where the first four elements of the last dimension
                represent (x_min, y_min, x_max, y_max).

    Returns:
        torch.Tensor: The bbox coordinates converted to center format.
                      Shape is (..., 4). All other elements in the last dimension are discarded.
    '''
    x_min = bboxes[..., 0]
    y_min = bboxes[..., 1]
    x_max = bboxes[..., 2]
    y_max = bboxes[..., 3]
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return torch.stack([x_center, y_center, width, height], dim = -1)

def center_to_corner_format(bboxes):
    '''
    Converts bbox coordinates from center (x_center, y_center, width, height) format
    to corner (x_min, y_min, x_max, y_max) format.

    In PyTorch BoundingBoxes convention: CXCYWH -> XYXY

    Args:
        bboxes: Bbox coordinates in center format.
                Shape is (..., 4+), where the first four elements of the last dimension
                represent (x_center, y_center, width, height).

    Returns:
        torch.Tensor: The bbox coordinates converted to corner format.
                      Shape is (..., 4). All other elements in the last dimension are discarded.
    '''
    x_center = bboxes[..., 0]
    y_center = bboxes[..., 1]
    width = bboxes[..., 2]
    height = bboxes[..., 3]
    
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    
    return torch.stack([x_min, y_min, x_max, y_max], dim = -1)

def yolov1_to_corner_format(bboxes: torch.Tensor, 
                            grid_i: torch.Tensor, 
                            grid_j: torch.Tensor, 
                            S: int = 7) -> torch.Tensor:
    '''
    Converts bboxes from YOLOv1 format to corner (x_center, y_center, width, height) format.
    In YOLOv1, each bounding box is predicted in a variation of center format:
        - (x_center, y_center) are relative to grid cell boundaries.
        - (width, height) are relative to the full image.

    Args:
        bboxes (torch.Tensor): The predicted bounding boxes returned by a YOLOv1 model.
                               The shape is (batch_size, S, S, B, 5), where the last dimension 
                               is in (x_center, y_center, width, height, confidence) format.
                               Note that (x_center, y_center) are relative to grid boundaries,
                               while (width, height) are relative to the full image.
        grid_i (torch.Tensor): Row indices for each grid cell, used to shift y_center to be relative to the full image.
                               Shape is (1, S, S, 1).
        grid_j (torch.Tensor): Column indices for each grid cell, used to shift y_center to be relative to the full image.
                               Shape is (1, S, S, 1).
        S (int): Grid size used for normalizing (x_center, y_center). Default is 7.

    Returns:
        torch.Tensor: Bbox coordinates in corner format, with confidence preserved.
                      Shape is (batch_size, S, S, B, 5), with
                      last dimension as (x_min, y_min, x_max, y_max, confidence).
    '''
    bboxes[..., 0] = bboxes[..., 0] + grid_j # Converting center_x coordinates
    bboxes[..., 1] = bboxes[..., 1] + grid_i # Converting center_y coordinates
    bboxes[..., 0:2] = bboxes[..., 0:2] / S

    # Convert from center format (CXCYWH) to corner format (XYXY) for the bbox coordinates
        # Shape: (batch_size, S, S, B, 4)
        # This gets rid of confidence scores
    corner_bboxes = center_to_corner_format(bboxes)

    # Concatenate back confidence scores to the last dimension
    return torch.concat([corner_bboxes, bboxes[..., 4:]], dim = -1)