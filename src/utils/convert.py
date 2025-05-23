#####################################
# Functions
#####################################
import torch


#####################################
# Functions
#####################################
def corner_to_center_format(bboxes):
    '''
    bboxes shape: (..., 4+) where the first four elements of the last dimension are XYXY
    XYXY -> CXCYWH
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
    bboxes shape: (..., 4+) where the first four elements of the last dimension are CXCYWH
    CXCYWH -> XYXY
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
    Args:
        bboxes (torch.Tensor): The predicted bounding boxes returned by a YOLOv1 model.
                               The shape is (batch_size, S, S, B, 5), where the last dimension 
                               is in (x_center, y_center, width, height, confidence) format.
                               Note that (x_center, y_center) are relative to grid boundaries,
                               while (width, height) are relative to the full image.
        grid_i shape: (1, S, S, 1)
        grid_j shape: (1, S, S, 1)
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