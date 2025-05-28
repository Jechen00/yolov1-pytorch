#####################################
# Imports & Dependencies
#####################################
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Optimizer

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Tuple

from src import engine, constants, evaluate
from src.utils import misc


#####################################
# Functions
#####################################
def yolov1_train_step(model: nn.Module, 
                      dataloader: DataLoader, 
                      loss_fn: nn.Module, 
                      optimizer: Optimizer, 
                      accum_steps: int = 1,
                      clip_grads: bool = False,
                      max_norm: float = 1.0,
                      device: Union[torch.device, str] = 'cpu') -> Dict[str, float]:
    '''
    loss_fn (nn.Module): Loss function used as the error metric.
        The reduction method for the loss function must be 'mean'.
        Also, it should return a loss dictionary storing the following components of the YOLOv1 loss:
             - total: The total loss, summing together all components of the YOLOv1 loss
             - class: The classification loss in object cells
             - local: The localization loss (center coordinates, height, and width) in object cells
             - obj_conf: The object confidence loss for responsible bboxes in object cells
             - noobj_conf: The no-object confidence loss for bboxes in no-object cells
                           and non-responsible bboxes in object cells
    accum_steps (int): Number of batches to loop over before performing an optimizer step.
                       If `accum_steps > 1`, gradients are accumulated over multiple batches,
                       simulating a larger batch size. Default is 1.
                       See: https://lightning.ai/blog/gradient-accumulation/
    clip_grads (bool): Controls whether gradient clipping should be used to prevent exploding gradients. Default is False.
    max_norm (float): Maximum norm for gradients, only used if gradient clipping is enabled. Default is 1.0.
    '''
    
    assert accum_steps > 0, 'Number of accumulation steps, `accum_steps`, must be at least 1'
    assert getattr(loss_fn, 'reduction', 'mean') == 'mean', "Expected 'mean' reduction in loss_fn"
    
    num_samps = len(dataloader.dataset)
    
    loss_sums = {
        'total': 0.0,
        'class': 0.0,
        'local': 0.0,
        'obj_conf': 0.0,
        'noobj_conf': 0.0
    }
    
    model.train()
    for i, (imgs, targs) in enumerate(dataloader):
        imgs, targs = imgs.to(device), targs.to(device)
        batch_size = imgs.shape[0]
        
        pred_logits = model(imgs)
        
        # Compute loss for the batch
        loss_dict = loss_fn(pred_logits, targs)
        (loss_dict['total'] / accum_steps).backward() # Backpropagate only through the total loss

        # Clip gradients if necessary
        if clip_grads:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_norm)

        # print(f'[BATCH {i}, GRADIENT]: {misc.calc_grad_norm(model, order = 2)}')

        for key in loss_sums:
            # Multiplying by batch_size gives 'sum' reduction
            loss_sums[key] += loss_dict[key].detach() * batch_size
            
        # Used to simulate a larger batch_size
        batch_idx = i + 1
        if (batch_idx % accum_steps == 0) or (batch_idx == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()
    
    # Divide loss_sums by num_samps to get accurate average over all samples
    return {key: loss_sums[key].item() / num_samps for key in loss_sums}


def train(model: nn.Module,
          train_loader: DataLoader, 
          test_loader: DataLoader, 
          loss_fn: nn.Module, 
          optimizer: Optimizer, 
          scheduler: lr_scheduler._LRScheduler, 
          te_cfgs: TrainEvalConfigs,
          ckpt_cfgs: CheckpointConfigs,
          device: Union[str, torch.device] = 'cpu') -> Tuple[dict, dict]:
    
    # -------------------------
    # Setup & Initialization
    # -------------------------
    start_logs = [] # Log messages to print prior to training/evaluation

    if ckpt_cfgs.save_path is not None:
        start_logs.append(
            f'{constants.BOLD_START}[NOTE]{constants.BOLD_END} ' 
            f'Checkpoints will be saved to {ckpt_cfgs.save_path}.'
        )
    
    # Load in previous checkpoint if resuming training
    if ckpt_cfgs.resume:
        checkpoint = torch.load(ckpt_cfgs.resume_path, map_location = device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        last_epoch = checkpoint['last_epoch']
        epoch_losses = checkpoint['epoch_losses']
        map_history = checkpoint['map_history']
            
        start_logs.append(
            f'{constants.BOLD_START}[NOTE]{constants.BOLD_END} '
            f'Successfully loaded previous checkpoint at {ckpt_cfgs.resume_path}. '
            f'Resuming training from epoch {last_epoch + 1}.'
        )

    else:
        last_epoch = -1
        epoch_losses = {
            'total': [],
            'class': [],
            'local': [],
            'obj_conf': [],
            'noobj_conf': []
        }
        map_history = {} # This is only used if eval_intervals is not None
    
    for log in start_logs:
        print(log)
    print()  # Add a blank line before training/evaluation logs

    # Start of training and evaluation
    for epoch in range(last_epoch + 1, te_cfgs.num_epochs):
        epoch_logs = [] # Log messages for the epoch
        
        # -------------------------
        # Training
        # -------------------------
        train_start = time.time()
        
        # Compute average losses (over batches) for the epoch
            # This returns a dictionary where the values are already floats
        avg_losses = engine.yolov1_train_step(model = model, dataloader = train_loader, 
                                              loss_fn = loss_fn, optimizer = optimizer, 
                                              accum_steps = te_cfgs.accum_steps, 
                                              clip_grads = te_cfgs.clip_grads,
                                              max_norm = te_cfgs.max_norm, 
                                              device = device)
        # Update optimizer learning rates
        scheduler.step()
        
        # Store each average loss
        for key in epoch_losses:
            epoch_losses[key].append(avg_losses[key])
        
        train_end = time.time()
        
        epoch_logs.append(
            f'{constants.BOLD_START}[EPOCH {epoch:>3} | {"Train Loss":<12}]{constants.BOLD_END} '
            f'Total: {avg_losses["total"]:<7.4f} | '
            f'Class: {avg_losses["class"]:<7.4f} | '
            f'Local: {avg_losses["local"]:<7.4f} | '
            f'ObjConf: {avg_losses["obj_conf"]:<7.4f} | '
            f'NoObjConf: {avg_losses["noobj_conf"]:<7.4f}'
        )
        
        train_time = f'{(train_end - train_start):.2f}' + ' sec'
        time_log = (
            f'{constants.BOLD_START}[EPOCH {epoch:>3} | {"Time":<12}]{constants.BOLD_END} '
            f'Train: {train_time:<11}'
        )
        
        # -------------------------
        # Evaluation
        # -------------------------
        # Evaluate mAP at specified intervals and at the final epoch
        should_eval = (te_cfgs.eval_intervals is not None) and (
            (epoch % te_cfgs.eval_intervals == 0) or (epoch == te_cfgs.num_epochs - 1)
        )
        if should_eval: 
            eval_start = time.time()

            map_dict = evaluate.calc_dataset_map(model = model, dataloader = test_loader, 
                                                obj_threshold = te_cfgs.obj_threshold,
                                                nms_threshold = te_cfgs.nms_threshold,
                                                map_thresholds = te_cfgs.map_thresholds,
                                                device = device)

            map_history[epoch] = {
                'map': map_dict['map'].item(),
                'map_50': map_dict['map_50'].item(), # -1 if 0.5 not in map_thresholds
                'map_75': map_dict['map_75'].item(), # -1 if 0.75 not in map_thresholds
                'map_per_class': map_dict['map_per_class'].tolist(),
                'classes': map_dict['classes'].tolist(),
                'obj_threshold': te_cfgs.obj_threshold,
                'nms_threshold': te_cfgs.nms_threshold,
                'map_thresholds': te_cfgs.map_thresholds
            }

            eval_end = time.time()

            epoch_logs.append(
                f'{constants.BOLD_START}[EPOCH {epoch:>3} | {"Eval Metrics":<12}]{constants.BOLD_END} '
                f'mAP: {map_history[epoch]["map"]:.4f}'
            )

            eval_time = f'{(eval_end - eval_start):.2f}' + ' sec'
            time_log += f' | Eval: {eval_time:<11}'
        
        # -------------------------
        # Saving and Logs
        # -------------------------
        if ckpt_cfgs.save_path is not None:
            misc.save_checkpoint(model = model, 
                                 optimizer = optimizer, 
                                 scheduler = scheduler,
                                 epoch_losses = epoch_losses,
                                 map_history = map_history,
                                 last_epoch = epoch,
                                 save_path = ckpt_cfgs.save_path)
        
        epoch_logs.append(time_log + '\n')
        for log in epoch_logs:
            print(log)
    
    return epoch_losses, map_history
    

#####################################
# Classes
#####################################
class WarmupMultiStepLR(lr_scheduler.MultiStepLR):
    '''
    This adds a warmup period to the MultiStepLR scheduler from: 
        https://github.com/pytorch/pytorch/blob/v2.7.0/torch/optim/lr_scheduler.py#L485

    optimizer (torch.optim.Optimizer): Optimizater whose learning rates will be changed by the scheduler.
    pre_warmup_lrs (List[float]): A list of learning rates for each parameter group 
                                  at the start of training (epoch 0).
                                  The scheduler will linearly increase the learning rate 
                                  from these values to the base learning rates over the warmup period.
    milestones (list): List of indices for the milestone epochs to apply learning rate decay.
                       These indices must be after the value of `warmup_epochs`.
    warmup_epochs (int): Number of epochs over which to linearly increase the learning rates from 
                         pre_warmup_lrs to the base learning rates. 
                         On epoch `warmup_epochs` learning rates will reach the base learning rates.
                         If `warmup_epochs = 0`, the behavior of the scheduler will be the same as MultiStepLR.
                         Default is 5.
    gamma (float): Multiplicative factor for the learning rate decay. Default is 0.1.
    last_epoch (int): The index of last epoch. Default is -1, which indicates the start of training.
    '''
    def __init__(self, 
                 optimizer: Optimizer, 
                 pre_warmup_lrs: List[float],
                 milestones: List[int], 
                 warmup_epochs: int = 5,
                 gamma: float = 0.1,
                 last_epoch: int = -1):
        
        invalid = [m for m in milestones if m <= warmup_epochs]
        assert not invalid, f'Milestones {invalid} must all be after `warmup_epochs` ({warmup_epochs})'
        
        assert len(pre_warmup_lrs) == len(optimizer.param_groups), (
            'Length of `pre_warmup_lrs` must match number of parameter groups in `optimizer`'
        )
        
        self.pre_warmup_lrs = pre_warmup_lrs
        self.warmup_epochs = warmup_epochs
        
        super().__init__(optimizer = optimizer, milestones = milestones, 
                         gamma = gamma, last_epoch = last_epoch) # Initializes self.base_lrs
          
    def get_lr(self):
        '''
        Returns the learning rate of each parameter group in `optimizer`.
        For the MultiStepLR scheduler component, please see:
            https://github.com/pytorch/pytorch/blob/v2.7.0/torch/optim/lr_scheduler.py#L522
        '''

        if (self.last_epoch <= self.warmup_epochs) and (self.warmup_epochs > 0):            
            # Warmup phase (Linearly changes pre_warmup_lrs to base_lrs)
                # Each warmup step is (b_lr - w_lr) / warmup_epochs
            return [
                w_lr +  self.last_epoch * (b_lr - w_lr) / self.warmup_epochs
                for w_lr, b_lr in zip(self.pre_warmup_lrs, self.base_lrs)
            ]
        else:
            return super().get_lr()
        
@dataclass
class TrainEvalConfigs():
    '''
    Class for setting YOLOv1 training and evaluation configurations.
    '''
    num_epochs: int
    accum_steps: int = 1
    clip_grads: bool = False
    max_norm: float = 1.0
        
    eval_intervals: Optional[int] = None
    obj_threshold: float = 0.2
    nms_threshold: float = 0.5
    map_thresholds: Optional[List[float]] = None
        
@dataclass
class CheckpointConfigs():
    '''
    Class for setting checkpoint saving and resuming configurations.
    '''
    save_dir: Optional[str] = None
    checkpoint_name: Optional[str] = None
    ignore_exists: bool = False
    resume_path: Optional[str] = None
    resume: bool = False
    
    def __post_init__(self):
        # Get the save_path
        match (self.save_dir, self.checkpoint_name):
            case (None, None):
                self.save_path = None # No saving needed

            case (str(), str()):
                # Add .pth if checkpoint_name doesn't end with .pth or .pts
                if not self.checkpoint_name.endswith(('.pth', '.pt')):
                    self.checkpoint_name += '.pth'
                self.save_path = os.path.join(self.save_dir, self.checkpoint_name)

            case (str(), None):
                # Set a default file name for saved checkpoint
                self.save_path =  os.path.join(self.save_dir, 'checkpoint.pth')

            case (None, str()):
                raise ValueError('`save_dir` must be a specified string if `checkpoint_name` is given.')
                
            
        # Check if resuming and if a resume_path needs to be set
        if self.resume and (self.resume_path is None):
            assert self.save_path is not None, (
                'Cannot resume training. Neither `resume_path` is provided, '
                'nor both `save_dir` and `checkpoint_name`.'
            )
            # Use save_path as resume_path if none was provided explicitly
            self.resume_path = self.save_path
        
        # If not resuming, check if save_path already has an existing file
        elif (not self.resume) and (self.save_path is not None):
            if os.path.exists(self.save_path) and (not self.ignore_exists):
                raise FileExistsError(
                    f'A file already exists at `save_path`: {self.save_path}, but `resume = False`. '
                    f'To allow overwriting this file and start training from scratch, set `ignore_exists = True`.'
                )