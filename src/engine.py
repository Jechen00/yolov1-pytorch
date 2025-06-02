#####################################
# Imports & Dependencies
#####################################
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Optimizer
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Tuple

from src import engine, constants, evaluate, postprocess
from src.utils import misc


#####################################
# Functions
#####################################
def yolov1_train_step(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_fn: nn.Module, 
    optimizer: Optimizer, 
    accum_steps: int = 1,
    clip_grads: bool = False,
    max_norm: float = 1.0,
    device: Union[torch.device, str] = 'cpu'
) -> Dict[str, float]:
    '''
    Performs a single training epoch for a YOLOv1 model, 
    including optional gradient accumulation and clipping.

    Args:
        model (nn.Module): The YOLOv1 model to train. Should be already on `device`.
        dataloader (Dataloader): Dataloader for the training dataset.
        loss_fn (nn.Module): YOLOv1 loss function with reduction method set to 'mean'. 
                             Its output should be a dictionary with keys matching `constants.LOSS_KEYS`, 
                             including a 'total' key representing the full loss.
        optimizer (Optimizer): Optimizer used to update model parameters every accumulated batch.
        accum_steps (int): Number of batches to loop over before performing an optimizer step.
                           If `accum_steps > 1`, gradients are accumulated over multiple batches,
                           simulating a larger batch size. Default is 1.
                           See: https://lightning.ai/blog/gradient-accumulation/
        clip_grads (bool): Whether gradient clipping should be used to prevent exploding gradients. Default is False.
        max_norm (float): Maximum norm for gradients, only used if gradient clipping is enabled. Default is 1.0.
        device (torch.device or str): The device to perform computations on. Default is 'cpu'.

    Returns:
        Dict[str, float]: Dictionary mapping the components of the YOLOv1 training loss
                          to its value averaged over all samples in the dataset. 
                          The keys of this dictionary are the same as the output of `loss_fn`.
    '''
    
    assert accum_steps > 0, 'Number of accumulation steps, `accum_steps`, must be at least 1'
    assert getattr(loss_fn, 'reduction', 'mean') == 'mean', "Expected 'mean' reduction in loss_fn"
    
    num_samps = len(dataloader.dataset)
    loss_sums = {key: 0.0 for key in constants.LOSS_KEYS}
    
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
    
def yolov1_val_step(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_fn: nn.Module, 
    should_eval: bool = False,
    obj_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    map_thresholds: Optional[List[float]] = None,
    device: Union[torch.device, str] = 'cpu'
) -> Tuple[dict, Optional[dict]]:
    '''
    Performs a single validation epoch for a YOLOv1 model. 
    This includes YOLOv1 loss computation and optional evaluation metrics (mAP and mAR).

    Args:
        model (nn.Module): The YOLOv1 model to train. Should be already on `device`.
        dataloader (Dataloader): Dataloader for the validation dataset.
        loss_fn (nn.Module): YOLOv1 loss function with reduction method set to 'mean'. 
                             Its output should be a dictionary with keys matching `constants.LOSS_KEYS`, 
                             including a 'total' key representing the full loss.
        should_eval (bool): Whether to compute evaluation metrics (mAP and mAR). Default is False.
                            If `True`, must provide `obj_threshold`, `nms_threshold`, and `map_thresholds`.
        obj_threshold (optional, float): Threshold to filter out low predicted object confidence scores 
                                         when computing mAP/mAR. 
        nms_threshold (optional, float): The IoU threshold used when performing NMS for mAP/mAR.
        map_thresholds (optional, List[float]): A list of IoU thresholds used for mAP/mAR calculations.
        device (torch.device or str): The device to perform computations on. Default is 'cpu'.

    Returns:
        loss_avgs (Dict[str, float]): Dictionary mapping the components of the YOLOv1 validation loss
                                      to its value averaged over all samples in the dataset. 
                                      The keys of this dictionary are the same as the output of `loss_fn`.

        eval_res (dict or None): If `should_eval = True`, this is a metric dictionary (with mAP and mAR values) 
                                 produced by `MeanAveragePrecision.compute()`. If `should_eval = False`, this is None

                                 For more details on the metric dictionaries, see:
                                        https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
    '''
    assert getattr(loss_fn, 'reduction', 'mean') == 'mean', "Expected 'mean' reduction in loss_fn"

    if should_eval:
        eval_res = {
            'obj_threshold': obj_threshold,
            'nms_threshold': nms_threshold,
            'map_thresholds': map_thresholds
        }
        for key, value in eval_res.items():
            assert value is not None, f'If `should_eval` is True, `{key}` must not be None'

        map_metric = MeanAveragePrecision(box_format = 'xyxy', 
                                          class_metrics = True, 
                                          iou_thresholds = map_thresholds)
        
    num_samps = len(dataloader.dataset)
    loss_sums = {key: 0.0 for key in constants.LOSS_KEYS}

    model.eval()
    for imgs, targs in dataloader:
        imgs, targs = imgs.to(device), targs.to(device)
        batch_size = imgs.shape[0]
        
        with torch.inference_mode():
            pred_logits = model(imgs)
        
        # -------------------------
        # Loss
        # -------------------------
        # Compute loss for the batch
        loss_dict = loss_fn(pred_logits, targs)

        for key in loss_sums:
            # Multiplying by batch_size gives 'sum' reduction
            loss_sums[key] += loss_dict[key] * batch_size

        # -------------------------
        # Evaluation Metrics
        # -------------------------
        if should_eval:
            targ_dicts = postprocess.decode_targets_yolov1(targs, S = model.S, B = model.B)
            pred_dicts = evaluate.predict_yolov1_from_logits(
                pred_logits = pred_logits,
                S = model.S, B = model.B,
                obj_threshold  = obj_threshold, 
                nms_threshold = nms_threshold
            )

            map_metric.update(pred_dicts, targ_dicts)

    loss_avgs = {key: loss_sums[key].item() / num_samps for key in loss_sums}
    if should_eval:
        map_res = map_metric.compute() # Compute mAP and mAR values
        for key, value in map_res.items():
            # Convert tensors to floats/lists
            eval_res[key] = value.item() if value.ndim == 0 else value.tolist()

        return loss_avgs, eval_res
    else:
        return loss_avgs, None

def train(
    model: nn.Module,
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    loss_fn: nn.Module, 
    optimizer: Optimizer, 
    scheduler: lr_scheduler._LRScheduler, 
    te_cfgs: TrainEvalConfigs,
    ckpt_cfgs: CheckpointConfigs,
    device: Union[str, torch.device] = 'cpu'
) -> Tuple[dict, dict, dict]:
    '''
    Trains a YOLOv1 model, tracking loss values and evaluation metrics (e.g. mAP and mAR).
    Supports training from scratch or resuming from a checkpoint.

    The flow of each epoch is as follows:
        - Computes training loss and updates the model (per accumulated batch)
        - Computes validation loss per epoch
        - Optionally computes mAP/mAR at evaluation epochs
        - Optionally saves model checkpoints

    Args:
        model (nn.Module): The YOLOv1 model to train. Should be already on `device`.
        train_loader (DataLoader): Dataloader for training dataset.
        val_loader (DataLoader):  Dataloader for validation dataset.
        loss_fn (nn.Module): The YOLOv1 loss function used as the error metric.
                             The reduction method for the loss function must be 'mean'.
                             It's output should also be a dictionary containing the keys in `constants.LOSS_KEYS`.
                             The most important key is 'total', representing the full YOLOv1 loss value.
        optimizer (Optimizer): Optimizer used to update model parameters every accumulated batch.
        scheduler (lr_scheduler._LRScheduler): Learning rate scheduler. 
        te_cfgs (TrainEvalConfigs): Configuration dataclass for training and evaluation parameters.
        ckpt_cfgs (CheckpointConfigs): Configuration dataclass for saving and resuming checkpoints.
        device (torch.device or str): The device to perform computations on. Default is 'cpu'.

    Returns:
        train_losses (Dict[str, list]): Dictionary mapping loss components in `constants.LOSS_KEYS`
                                        to their list of training values per epoch.
        val_losses (Dict[str, list]): Same as `train_losses`, but for the validation dataset.
        eval_history (dict): Dictionary mapping evaluation epoch indices (int)
                             to metric dictionaries (with mAP and mAR values) 
                             returned by `MeanAveragePrecision.compute()`. 

                            Example eval_history: 
                                {
                                    5: {'map': 0.45, ...},
                                    10: {'map': 0.5, ...}
                                }

                            For more details on the metric dictionaries, see:
                                https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
    '''
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
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        eval_history = checkpoint['eval_history']
            
        start_logs.append(
            f'{constants.BOLD_START}[NOTE]{constants.BOLD_END} '
            f'Successfully loaded checkpoint at {ckpt_cfgs.resume_path}. '
            f'Resuming training from epoch {last_epoch + 1}.'
        )

    else:
        last_epoch = -1
        train_losses = {key: [] for key in constants.LOSS_KEYS}
        val_losses = {key: [] for key in constants.LOSS_KEYS}
        eval_history = {} # This is only used if eval_intervals is not None
    
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

        # Compute average losses (over batches)
        train_avgs = engine.yolov1_train_step(model = model, dataloader = train_loader, 
                                              loss_fn = loss_fn, optimizer = optimizer, 
                                              accum_steps = te_cfgs.accum_steps, 
                                              clip_grads = te_cfgs.clip_grads,
                                              max_norm = te_cfgs.max_norm, 
                                              device = device)
        # Update optimizer learning rates
        scheduler.step()
        
        # Store and log each average loss
        train_log = f'{constants.BOLD_START}[EPOCH {epoch:>3} | {"Train Loss":<12}]{constants.BOLD_END} '
        for key in constants.LOSS_KEYS:
            train_losses[key].append(train_avgs[key])
            train_log += f'{constants.LOSS_NAMES[key]}: {train_avgs[key]:<7.4f} | '
        
        epoch_logs.append(train_log)

        train_end = time.time()

        train_time = f'{(train_end - train_start):.2f}' + ' sec'
        time_log = (
            f'{constants.BOLD_START}[EPOCH {epoch:>3} | {"Time":<12}]{constants.BOLD_END} '
            f'Train: {train_time:<11}'
        )

        # -------------------------
        # Validation
        # -------------------------
        val_start = time.time()

        # Evaluate metrics (mAP) at specified intervals and at the final epoch
        should_eval = (te_cfgs.eval_intervals is not None) and (
            (epoch % te_cfgs.eval_intervals == 0) or (epoch == te_cfgs.num_epochs - 1)
        )

        # Compute average losses (over batches) and eval metrics
            # eval_res is None if should_eval is False
        val_avgs, eval_res = yolov1_val_step(model = model, dataloader = val_loader,
                                             loss_fn = loss_fn,
                                             should_eval = should_eval,
                                             obj_threshold = te_cfgs.obj_threshold,
                                             nms_threshold = te_cfgs.nms_threshold,
                                             map_thresholds = te_cfgs.map_thresholds,
                                             device = device)
        # Store and log each average loss
        val_log = f'{constants.BOLD_START}[EPOCH {epoch:>3} | {"Val Loss":<12}]{constants.BOLD_END} '
        for key in constants.LOSS_KEYS:
            val_losses[key].append(val_avgs[key])
            val_log += f'{constants.LOSS_NAMES[key]}: {val_avgs[key]:<7.4f} | '

        epoch_logs.append(val_log)

        # Store and log eval metrics
        if should_eval:
            eval_history[epoch] = eval_res

            epoch_logs.append(
                f'{constants.BOLD_START}[EPOCH {epoch:>3} | {"Val Metrics":<12}]{constants.BOLD_END} '
                f'mAP: {eval_res["map"]:.4f}'
            )
        val_end = time.time()

        val_time = f'{(val_end - val_start):.2f}' + ' sec'
        time_log += f' | Val: {val_time:<11}'
        
        # -------------------------
        # Saving and Logs
        # -------------------------
        if ckpt_cfgs.save_path is not None:
            misc.save_checkpoint(model = model, 
                                 optimizer = optimizer, 
                                 scheduler = scheduler,
                                 train_losses = train_losses,
                                 val_losses = val_losses,
                                 eval_history = eval_history,
                                 last_epoch = epoch,
                                 save_path = ckpt_cfgs.save_path)
        
        epoch_logs.append(time_log + '\n')
        for log in epoch_logs:
            print(log)
    
    return train_losses, val_losses, eval_history


#####################################
# Classes
#####################################
class WarmupMultiStepLR(lr_scheduler.MultiStepLR):
    '''
    This adds a warmup period to the MultiStepLR scheduler from: 
        https://github.com/pytorch/pytorch/blob/v2.7.0/torch/optim/lr_scheduler.py#L485
    
    Args:
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
    Data class for setting YOLOv1 training and evaluation configurations.

    Args:
        num_epochs (int): Number of epochs to train the YOLOv1 model.
        accum_steps (int): Number of batches to loop over before updating model parameters. 
                           Applies during training only. 
                           If `accum_steps > 1`, gradients are accumulated over multiple batches,
                           simulating a larger batch size. Default is 1.
                           See: https://lightning.ai/blog/gradient-accumulation/
        clip_grads (bool): Whether to clip gradients during training to prevent exploding gradients. Default is False.
        max_norm (float): Maximum norm for gradients, only used if gradient clipping is enabled. Default is 1.0.
        eval_intervals (optional, int): Number of epochs to wait before computing evaluation metrics on the validation dataset.
                                        If None, evaluation metrics are never computed. Default is None.
        obj_threshold (float): Threshold to filter out low predicted object confidence scores. 
                               Used during evaluation when computing mAP/mAR. Default is 0.25.
        nms_threshold (float): The IoU threshold used during evaluation when performing NMS for mAP/mAR. Default is 0.5.
        map_thresholds (optional, List[float]): A list of IoU thresholds used for mAP/mAR calculations.
                                                If not provided, this defaults to [0.5].
    '''
    num_epochs: int
    accum_steps: int = 1
    clip_grads: bool = False
    max_norm: float = 1.0
        
    eval_intervals: Optional[int] = None
    obj_threshold: float = 0.2
    nms_threshold: float = 0.5
    map_thresholds: Optional[List[float]] = None

    def __post_init__(self):
        # Set a default value for map_thresholds
        if self.eval_intervals is not None:
            self.map_thresholds = [0.5] if self.map_thresholds is None else self.map_thresholds
        
@dataclass
class CheckpointConfigs():
    '''
    Data class for setting checkpoint saving and resuming configurations.

    Args:
        save_dir (optional, str): Directory to save checkpoint every epoch.
                                  Required if `checkpoint_name` is provided.
                                  If `save_dir` and `checkpoint_name` are None, checkpoints will not be saved.
        checkpoint_name (optional, str): File name for the checkpoint. 
                                         If missing an extension (.pt or .pth), `.pth` will be appended.
                                         If only `save_dir` is provided, defaults to `checkpoint.pth`.
        ignore_exists (bool): Whether to ignore existing checkpoint file at `save_dir/checkpoint_name`.
                              If `False` and a file already exists, training is halted unless `resume = True`.
        resume_path (optional, str): Full path to a checkpoint file to resume training from.
                                     If not provided and `resume = True`, defaults to `save_dir/checkpoint_name`.
        resume (bool): Whether to resume training from a previous checkpoint.
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
            if os.path.isfile(self.save_path) and (not self.ignore_exists):
                raise FileExistsError(
                    f'A file already exists at `save_path`: {self.save_path}, but `resume = False`. '
                    f'To allow overwriting this file and start training from scratch, set `ignore_exists = True`.'
                )