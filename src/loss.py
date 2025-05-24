#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

from src import evaluate, postprocess
from src.utils import convert


#####################################
# Loss Class
#####################################
class YOLOv1Loss(nn.Module):
    '''
    Loss function for YOLOv1 model.
    Reference: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf
    '''
    def __init__(self, S: int = 7, B: int = 2, C: int = 20,
                 lambda_coord: float = 5, lambda_noobj: float = 0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord, self.lambda_noobj = lambda_coord, lambda_noobj
    
    def class_loss(self, 
                   preds: torch.Tensor, 
                   targs: torch.Tensor, 
                   obj_mask: torch.Tensor) -> torch.Tensor:
        '''
        preds shape: (batch_size, S, S, B*5 + C)
        targs shape: (batch_size, S, S, B*5 + C)
        obj_mask shape: (batch_size, S, S)
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
        preds shape: (batch_size, S, S, B*5 + C)
        targs shape: (batch_size, S, S, B*5 + C)
        iou shape: (batch_size, S, S, B)
        obj_resp_mask shape: (batch_size, S, S, B)
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
        preds shape: (batch_size, S, S, B*5 + C)
        obj_resp_mask shape: (batch_size, S, S, B)
        '''
        
        # Note: By DeMorgan's Law, `~obj_resp_mask` is gives all bboxes that are either from no-object cells 
            # or are from object cells, but are not responsible for it
        # Note: I am including the predicted bboxes that are not responsible in an object cell
            # This is to specialize them so that if they are poor fits for the object cell (low IoU), 
            # their confidence should close to 0 and they should stay quiet.
        # Shape: (num_zero_confidence_bboxes, 5)
        pred_bboxes = preds[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)[~obj_resp_mask]

        return (pred_bboxes[..., -1]**2).sum()

    def forward(self, 
                preds: torch.Tensor, 
                targs: torch.Tensor) -> torch.Tensor:
        '''
        preds shape: (batch_size, S, S, B*5 + C)
        targs shape: (batch_size, S, S, B*5 + C)
        '''

        assert not torch.isnan(preds).any(), 'NaNs in preds'
        assert not torch.isnan(targs).any(), 'NaNs in targs'

        device = targs.device

        batch_size = targs.shape[0]

        grid_i, grid_j = torch.meshgrid(torch.arange(self.S), torch.arange(self.S), indexing = 'ij')
        grid_i = grid_i[None, ..., None].to(device)  # Shape: (1, S, S, 1)
        grid_j = grid_j[None, ..., None].to(device)  # Shape: (1, S, S, 1)

        # Shape: (batch_size, S, S, B*5 + C)
        preds = postprocess.decode_logits_yolov1(preds, self.S, self.B, split_output = False)

        # Shape: (batch_size, S, S, B, 5)
        targ_bboxes = targs[..., :self.B*5].clone().view(batch_size, self.S, self.S, self.B, 5)
        pred_bboxes = preds[..., :self.B*5].clone().view(batch_size, self.S, self.S, self.B, 5)

        # Convert center (x, y) coordinates from relative offsets within grid cell to relative within the full image
            # Note: coordinates are still normalized to the image width and height
            # Shape: (batch_size, S, S, B, 5)
        targ_bboxes = convert.yolov1_to_corner_format(targ_bboxes, grid_i, grid_j, self.S)
        pred_bboxes = convert.yolov1_to_corner_format(pred_bboxes, grid_i, grid_j, self.S)

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
        }

        # Divide by batch size to get mean
        for key in loss_dict:
            loss_dict[key] = loss_dict[key].detach().item() / batch_size

        loss_dict['total'] = tot_loss / batch_size # Not detached b/c will be used for backpropagation

        return loss_dict