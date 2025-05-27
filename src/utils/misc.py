#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

import os
import numpy as np
import random 
from typing import Dict, Optional


#####################################
# Functions
#####################################
def set_seed(seed: int = 0):
    '''
    Sets random seed and deterministic settings 
    for reproducibility across:
        - PyTorch
        - NumPy
        - Python's random module
        - CUDA
    
    Args:
        seed (int): The seed value to set.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def get_save_path(save_dir: Optional[str] = None, 
                  checkpoint_name: Optional[str] = None) -> Optional[str]:
    match (save_dir, checkpoint_name):
        case (None, None):
            return None # No saving needed

        case (str(), str()):
            # Add .pth if checkpoint_name doesn't end with .pth or .pts
            if not checkpoint_name.endswith(('.pth', '.pt')):
                checkpoint_name += '.pth'
            return os.path.join(save_dir, checkpoint_name)
        
        case (str(), None):
            # Set a default file name for saved checkpoint
            return os.path.join(save_dir, 'checkpoint.pth')
        
        case (None, str()):
            raise ValueError('`save_dir` must be a specified string if `checkpoint_name` is given.')
        


    if (save_dir is None) and (checkpoint_name is None):
        return None
    if save_dir is None:
        raise ValueError('`save_dir` must be specified if `checkpoint_name` is given.')
    if checkpoint_name is None:
        checkpoint_name = 'checkpoint.pth'
    if not checkpoint_name.endswith(('.pth', '.pt')):
        checkpoint_name += '.pth'
    return os.path.join(save_dir, checkpoint_name)

def save_checkpoint(model: nn.Module, 
                    optimizer: Optimizer, 
                    scheduler: lr_scheduler._LRScheduler, 
                    epoch_losses: Dict[str, list],
                    map_history: Dict[int, list],
                    last_epoch: int,
                    save_dir: Optional[str] = None, 
                    checkpoint_name: Optional[str] = None,
                    save_path: Optional[str] = None):
    '''
    Saves a checkpoint containing the model, optimizer, scheduler state dicts,
    along with training metrics and epoch index.

    Args:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer used during training.
        scheduler (lr_scheduler._LRScheduler): Learning rate scheduler.
        epoch_losses (Dict[str, list]): Dictionary of lists storing loss values per epoch.
        map_history (Dict[int, list]): Dictionary tracking evaluation metrics.
        last_epoch (int): Index of the last completed epoch.
        save_dir (Optional[str]): Directory to save the checkpoint.
        checkpoint_name (Optional[str]): Filename for the checkpoint (should end with '.pth' or '.pt').
        save_path (Optional[str]): Full path to save the checkpoint. 
                                   If provided, `save_dir` and `checkpoint_name` are ignored.

    '''
    if save_path is None:
        assert (save_dir is not None) and (checkpoint_name is not None), (
            'If `save_path` is not provided, both `save_dir` and `checkpoint_name` must be provided.'
        )
        # Create save path
        save_path = os.path.join(save_dir, checkpoint_name)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok = True)

    # Create checkpoint dictionary
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch_losses': epoch_losses,
        'map_history': map_history,
        'last_epoch': last_epoch
    }

    torch.save(obj = checkpoint, f = save_path)