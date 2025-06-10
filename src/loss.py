#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

from typing import Dict

from src import evaluate, postprocess
from src.utils import convert


#####################################
# Loss Class
#####################################
class YOLOv1Loss(nn.Module):
    '''
    Loss function for YOLOv1 model.
    Reference: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf

    Args:
        S (int): Grid size. Default is 7.
        B (int): Number of bounding boxes per grid cell. Default is 2.
        C (int): Number of classes. Default is 20.
        lambda_coord (float): Weight of the localization loss. Default is 5.
        lambda_noobj (float): Weight of the no-object loss. Default is 0.5.
        reduction ('sum' or 'mean'): Reduction method for the losses over batch samples.
                                     If 'sum' the losses are added. If 'mean', the losses are averaged.
                                     Default is 'mean'.
    '''
    def __init__(self, S: int = 7, B: int = 2, C: int = 20,
                 lambda_coord: float = 5, lambda_noobj: float = 0.5,
                 reduction: str = 'mean'):
        super().__init__()
        assert reduction in ['mean', 'sum'], "The `reduction` method must be 'mean' or 'sum'."

        self.S, self.B, self.C = S, B, C
        self.lambda_coord, self.lambda_noobj = lambda_coord, lambda_noobj
        self.reduction = reduction

    def class_loss(self, 
                   preds: torch.Tensor, 
                   targs: torch.Tensor, 
                   obj_mask: torch.Tensor) -> torch.Tensor:
        '''
        Computes the classification loss for the YOLOv1 model, defined as the 
        squared error between predicted class probabilities and target labels.
        This loss is only computed for grid cells with an object.

        Args:
            preds (torch.Tensor): Postprocessed model outputs of shape (batch_size, S, S, B*5 + C).
                                  The last C elements per cell should already be softmaxed.
            targs (torch.Tensor): Encoded target tensor representing the ground truth.
                                  Shape is (batch_size, S, S, B*5 + C).
            obj_mask (torch.Tensor): Boolean mask of shape (batch_size, S, S),
                                     indicating cells with an object.
        Returns:
            torch.Tensor: Scalar classification loss.
        '''
    
        # preds[obj_mask] shape: (num_object_cells, B*5 + C)
        # targs[obj_mask] shape: (num_object_cells, B*5 + C)
        prob_errors = (preds[obj_mask][:, -self.C:] - targs[obj_mask][:, -self.C:])**2
        return prob_errors.sum()

    def object_loss(self, 
                    preds: torch.Tensor, 
                    targs: torch.Tensor, 
                    ious: torch.Tensor, 
                    obj_resp_mask: torch.Tensor) -> torch.Tensor:
        '''
        Computes the localization loss and object confidence loss for the YOLOv1 model.
        This is only performed on responsible bboxes in grid cells with an object.

            - Localization Loss: The squared error between predicted and target bbox coordinates
                                 in (x_center, y_center, width, height) format.
                                 Note that (x_center, y_center) are relative to grid boundaries,
                                 while (width, height) are relative to the full image.
                                 The width and height are also square rooted before computing the loss.

            - Object Confidence Loss: The squared error between predicted object confidence 
                                      and IoU (target object confidence).

        Args:
            preds (torch.Tensor): Postprocessed model outputs of shape (batch_size, S, S, B*5 + C).
                                  The first B*5 elements per cell should already be sigmoided.
            targs (torch.Tensor): Encoded target tensor representing the ground truth.
                                  Shape is (batch_size, S, S, B*5 + C).
            ious: The IoU values computed between predicted and target bboxes. Shape is (batch_size, S, S, B).
            obj_resp_mask: Boolean mask of shape (batch_size, S, S, B), indicating responsible bboxes.

        Returns:
            torch.Tensor: Scalar localization loss.
            torch.Tensor: Scalar object confidence loss.
        '''
        
        # Note: this flattens across the batch and spatial dimensions
        # Shape: (num_responsible_bboxes, 5)
        pred_bboxes = preds[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)[obj_resp_mask]
        targ_bboxes = targs[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)[obj_resp_mask]
        
        coord_errors = (pred_bboxes[:, 0:2] - targ_bboxes[:, 0:2])**2
        length_errors = (pred_bboxes[:, 2:4].sqrt() - targ_bboxes[:, 2:4].sqrt())**2

        conf_errors = (pred_bboxes[:, -1] - ious[obj_resp_mask])**2
        
        # Localization error, Object confidence error
        return (coord_errors + length_errors).sum(), conf_errors.sum()

    def no_object_loss(self,
                       preds: torch.Tensor, 
                       obj_resp_mask: torch.Tensor) -> torch.Tensor:
        '''
        Computes the no-object confidence loss for the YOLOv1 model,
        defined as the squared error between predicted object confidence 
        and zero (target object confidence).

        This is computed for the following bboxes:
            - All bboxes (total is B) in grid cells without an object.
            - Non-responsible bboxes (total is B-1) in grid cells with an object.
        
        For the second point, these bboxes are included to specialize them,
        so that if they are poor fits for the object cell (low IoU), 
        their confidence should close to 0 and they should stay quiet.

        Args:
            preds (torch.Tensor): Postprocessed model outputs of shape (batch_size, S, S, B*5 + C).
                                  The first B*5 elements per cell should already be sigmoided.
            obj_resp_mask: Boolean mask of shape (batch_size, S, S, B), indicating responsible bboxes.
                        Note: By DeMorgan's Law, `~obj_resp_mask` gives all bboxes that are either from no-object cells 
                                or are from object cells, but are not responsible for it.

        Returns:
            torch.Tensor: Scalar no-object confidence loss.
        '''
        # Shape: (num_zero_confidence_bboxes, 5)
        pred_bboxes = preds[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)[~obj_resp_mask]

        return (pred_bboxes[..., -1]**2).sum()

    def forward(self, 
                pred_logits: torch.Tensor, 
                targs: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        Computes the full YOLOv1 loss across a batch, according to: https://arxiv.org/pdf/1506.02640.
        The components of this loss are also returned.

        Args:
            pred_logits (torch.Tensor): Output logits from a YOLOv1 model. 
                                        Shape is (batch_size, S, S, B*5 + C).
            targs (Torch.Tensor): Encoded target tensor representing the ground truth. 
                                  Shape is (batch_size, S, S, B*5 + C).

        Returns:
            loss_dict (Dict[str, torch.Tensor]): Dictionary containing the loss component values.
                The keys are as follows:
                    - total: The total loss, summing together all components of the YOLOv1 loss.
                    - class: The classification loss in object cells.
                    - local: The localization loss for responsible bboxes in object cells.
                    - obj_conf: The object confidence loss for responsible bboxes in object cells.
                    - noobj_conf: The no-object confidence loss for bboxes in no-object cells.
                                 and non-responsible bboxes in object cells.
        '''

        assert not torch.isnan(pred_logits).any(), 'NaNs in `pred_logits`'
        assert not torch.isnan(targs).any(), 'NaNs in `targs`'

        device = targs.device

        batch_size = targs.shape[0]

        grid_i, grid_j = torch.meshgrid(torch.arange(self.S), torch.arange(self.S), indexing = 'ij')
        grid_i = grid_i[None, ..., None].to(device)  # Shape: (1, S, S, 1)
        grid_j = grid_j[None, ..., None].to(device)  # Shape: (1, S, S, 1)

        # Shape: (batch_size, S, S, B*5 + C)
        preds = postprocess.activate_yolov1_logits(pred_logits, self.S, self.B, split_output = False)

        # Shape: (batch_size, S, S, B, 5)
        targ_bboxes = targs[..., :self.B*5].clone().view(batch_size, self.S, self.S, self.B, 5)
        pred_bboxes = preds[..., :self.B*5].clone().view(batch_size, self.S, self.S, self.B, 5)

        # Convert center (x, y) coordinates from relative offsets within grid cell to relative within the full image
            # Note: coordinates are still normalized to the image width and height
            # Shape: (batch_size, S, S, B, 5)
        targ_bboxes = postprocess.decode_yolov1_bboxes(targ_bboxes, grid_i, grid_j, self.S)
        pred_bboxes = postprocess.decode_yolov1_bboxes(pred_bboxes, grid_i, grid_j, self.S)

        # Note: this calculates IoUs for all grid cells, not just the ones with objects
            # This wastes computations, but allows for batch vectorization
            # For the last 2 dims, [i, j] gives IoU of i-th predicted bbox and j-th target bbox
        # Note: For targets, only the first bbox has meaningful non-zero values for object cells
        ious = evaluate.calc_ious(pred_bboxes, targ_bboxes)[..., 0] # Shape: (batch_size, S, S, B)

        # Assign predicted bbox responsibility based on max IoU
        resp_idx = ious.argmax(dim = -1)
        resp_mask = torch.nn.functional.one_hot(resp_idx, num_classes = self.B).bool() # Shape: (batch_size, S, S, B)

        obj_mask = targs[..., 4].bool() # Shape: (batch_size, S, S)
        obj_resp_mask = resp_mask & obj_mask.unsqueeze(-1) # Shape: (batch_size, S, S, B)
        
        class_loss = self.class_loss(preds, targs, obj_mask)
        local_loss, obj_conf_loss = self.object_loss(preds, targs, ious, obj_resp_mask)
        noobj_conf_loss = self.no_object_loss(preds, obj_resp_mask)

        tot_loss = (self.lambda_coord * local_loss +
                    obj_conf_loss +
                    self.lambda_noobj * noobj_conf_loss +
                    class_loss)

        loss_dict = {
            'class': class_loss,
            'local': local_loss,
            'obj_conf': obj_conf_loss,
            'noobj_conf': noobj_conf_loss,
            'total': tot_loss
        }

        if self.reduction == 'mean':
            # Divide by batch size to get mean
            for key in loss_dict:
                loss_dict[key] = loss_dict[key] / batch_size

        return loss_dict