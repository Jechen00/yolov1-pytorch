#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba
from matplotlib import patches
from matplotlib.axes import Axes
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Callable, Union

from src import evaluate, postprocess
from src.constants import VOC_PLOT_DISPLAYS, VOC_CLASSES, LOSS_NAMES, EVAL_NAMES


#####################################
# Functions
#####################################
def plot_results(train_losses: Dict[str, list], 
                 val_losses: Optional[Dict[str, list]] = None, 
                 eval_history: Optional[dict] = None,
                 loss_key: str = 'total',
                 eval_key: str = 'map') -> Figure:
    '''
    Plots training and evaluation metrics for a YOLOv1 model.   
    This function generates a matplotlib figure showing:
        - Training loss (required)
        - Validation loss (optional)
        - Evaluation metrics, such as mAP, for validation (optional)

    Args:
        train_losses (dict): A dictionary mapping loss component names to their
                             corresponding lists of training loss values per epoch.
                             This must include the key `loss_key`.
        val_losses (optional, dict): A dictionary mapping loss component names to their
                                     corresponding lists of validation loss values per epoch.
                                     This must include the key `loss_key`.
                                     Lists also must be the same length as those in `train losses`.
        eval_history (optional, dict): Dictionary mapping evaluation epoch indices (int)
                                       to dictionaries containing validation metrics (float).
                                       Each dictionary must include the key `eval_key`.

                                       Example for `eval_key = 'map'`: 
                                            {
                                                5: {'map': 0.45, ...},
                                                10: {'map': 0.5, ...}
                                            }
        loss_key (str): The loss component to plot from `train_losses` and `val_losses`.
                        Default is 'total' for the full YOLOv1 loss.
        eval_key (str): The evaluation metric key to extract and plot from `eval_history`.
                        Default is 'map' for mean Average Precision.

    Returns:
        Figure: A matplotlib figure with:
                    - One subplot (loss only) if `eval_history` is None.
                    - Two subplots (loss and eval metric) if `eval_history` is provided.
    '''
    train_loss = train_losses[loss_key]
    loss_epochs = np.arange(len(train_loss))
    
    if eval_history:
        eval_epochs = list(eval_history.keys())
        eval_scores = [eval_history[key][eval_key] for key in eval_history]
        ncols = 2
        figsize = (16, 8)
    else:
        ncols = 1
        figsize = (10, 8)

    fig, axes = plt.subplots(nrows = 1, ncols = ncols, figsize = figsize)
    if ncols == 1:
        axes = [axes]
    
    # Subplot 1: Losses
    loss_name = LOSS_NAMES.get(loss_key, loss_key)
    axes[0].plot(loss_epochs, train_loss, label = f'Train Loss: {loss_name}')
    if val_losses is not None:
        axes[0].plot(loss_epochs, val_losses[loss_key], label = f'Val Loss: {loss_name}')

    axes[0].set_ylabel('YOLOv1 Loss', fontsize = 24)

    # Subplot 2: mAP
    eval_name = EVAL_NAMES.get(eval_key, eval_key)
    if eval_history:
        axes[1].plot(eval_epochs, eval_scores, label = f'Val metric: {eval_name}', color = 'red')
        axes[1].yaxis.set_label_position('right')
        axes[1].yaxis.tick_right()

        axes[1].set_ylabel('Evaluation Metric', fontsize = 24,
                           rotation = 270, labelpad = 30)

    for ax in axes:
        ax.legend(fontsize = 20)
        ax.set_xlabel('Epoch', fontsize = 24)
        ax.grid(alpha = 0.5)

    fig.subplots_adjust(wspace = 0.04)
    plt.close(fig)
    
    return fig

def draw_bboxes(img: Image.Image, 
                bboxes: torch.Tensor, 
                labels: torch.Tensor, 
                scores: Optional[torch.Tensor] = None, 
                img_resize: Tuple[int, int] = (448, 448), 
                show_scores: bool = False,
                ax: Optional[Axes] = None,
                **kwargs) -> Optional[Figure]:
    '''
    Plots a PIL image along with a given set of bounding boxes (with labels + scores).

    Args:
        img (Image.Image): The PIL image to plot.
        bboxes (torch.Tensor): A tensor of bounding boxes in (x_min, y_min, x_max, y_max) format.
                               (x_min, y_min, x_max, y_max) should be on the same scale as `img_resize`.
                               Shape is (num_bboxes, 4).
        labels (torch.Tensor): A tensor of class labels for the bounding boxes in `bboxes`.
                               Shape is (num_bboxes,).
        scores (optional, torch.Tensor): A tensor of class confidence scores for the bounding boxes in `bboxes`.
                                         This is optional, but required if `show_scores` is True. Shape is (num_bboxes,).
        img_resize (Tuple[int, int]): The dimensions (width, height) the PIL image should be resized to prior to plotting.
                                      This should be the same scale that the bbox coordinates are set to. 
                                      Default is (448, 448) for YOLOv1. 
        show_scores (bool): Whether class confidence scores should be plotted along with the labels and bounding boxes.
                            If True, the `scores` argument is required. Default is False.
        ax (optional, Axes): A matplotlib axis to plot the image and bounding boxes. 
                             If this is provided, a figure will not be returned.
        **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.figure`.
                  This is only used if `ax` is not provided.

    Returns:
        fig (optional, Figure): A matplotlib figure with the plotted image and bounding boxes.
                                This is only returned if `ax` is not provided.
    '''
    if show_scores:
        assert scores is not None, 'A tensor of `scores` is required if `show_scores` is set to True.'
        scores = scores.cpu() # Send tensors to CPU, just in case they weren't there already
    bboxes, labels = bboxes.cpu(), labels.cpu()

    if ax is None:
        if not kwargs:
            kwargs = {'figsize': (10, 10)}
        fig = plt.figure(**kwargs)
        ax = plt.gca()
    else:
        fig = None
    
    ax.imshow(img.resize(img_resize)) # Plot resized image
    
    bbox_lw = 2.5
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        label_idx = labels[i]
        name, clr = VOC_PLOT_DISPLAYS[VOC_CLASSES[label_idx]]
        
        xmin, ymin, xmax, ymax = bbox
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin

        txt_x = xmin + bbox_lw
        txt_y = ymin + bbox_lw
        
        # Plot bbox
        rect = patches.Rectangle((xmin, ymin), bbox_w, bbox_h, 
                                 linewidth = bbox_lw, edgecolor = clr, facecolor = 'none')
        ax.add_patch(rect)
        
        if show_scores:
            txt = f'{name} \n Score: {scores[i]:.2f}'
        else:
            txt = f'{name}'
            
        # Annotate with the class name and class confidence score (if applicable)
        ax.text(txt_x, txt_y, txt,
                fontsize = 11, color = 'k',
                ha = 'left', va = 'top',
                bbox = dict(facecolor = clr, alpha = 1, pad = 2, edgecolor = 'none'))
        
    ax.axis(False)

    if fig is not None:
        plt.close(fig)
        return fig

def draw_label_grid(img: Image.Image, 
                    grid_labels: torch.Tensor,
                    grid_probs: Optional[torch.Tensor] = None,
                    S: int = 7,
                    img_resize: Tuple[int, int] = (448, 448),
                    show_probs: bool = False,
                    ax: Optional[Axes] = None,
                    **kwargs) -> Optional[Figure]:
    '''
    Overlays a grid on an image and annotates each cell with its predicted class label 
    and (optionally) the associated class probability.
    This essentially recreates the class probability map from Figure 2 of the YOLOv1 paper:
        https://arxiv.org/pdf/1506.02640

    Args:
        img (Image.Image): The PIL image to plot.
        grid_labels (torch.Tensor): Tensor of shape (S, S), where each value is an integer
                                    representing the predicted class label for the grid cell.
        grid_probs (Optional[torch.Tensor]): Tensor of shape (S, S), where each value is the
                                             predicted probability corresponding to the class in `grid_labels`.
                                             Required if `show_probs` is True.
        S (int): Grid size. Default is 7.
        img_resize (Tuple[int, int]): The dimensions (width, height) the PIL image should be resized to
                                      prior to plotting. Default is (448, 448) for YOLOv1. 
        show_probs (bool): Whether class probabilities from `grid_probs` should be annotated in each grid cell.
                           If `true`, `grid_probs` is required. Default is False.
        ax (optional, Axes): A matplotlib axis to plot the image and class label grid. 
                             If not provided, a new figure is created and returned.
        **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.figure`.
                  This is only used if `ax` is not provided.

    Returns:
        fig (optional, Figure): A matplotlib figure with the plotted image and class label grid.
                                This is only returned if `ax` is not provided.
    '''
    if show_probs:
        assert grid_probs is not None, 'A tensor of `grid_probs` is required if `show_probs` is set to True.'
        grid_probs = grid_probs.cpu() # Send tensors to CPU, just in case they weren't there already 
    grid_labels = grid_labels.cpu()

    if ax is None:
        if not kwargs:
            kwargs = {'figsize': (10, 10)}
        fig = plt.figure(**kwargs)
        ax = plt.gca()
    else:
        fig = None
    
    img = img.resize(img_resize)
    img_w, img_h = img.size
    
    # Height and width of a grid cell
    cell_h = img_h / S
    cell_w = img_w / S

    ax.imshow(img) # Plot resized image
    
   # Loop over rows
    for i in range(S):
        # Loop over columns
        for j in range(S):
            label = VOC_CLASSES[grid_labels[i, j]]
            name, cell_clr = VOC_PLOT_DISPLAYS[label]
            cell_clr_alpha = to_rgba(cell_clr, alpha = 0.4)

            # Plot grid cell
            cell_xmin = j * cell_w
            cell_ymin = i * cell_h
            cell = patches.Rectangle((cell_xmin, cell_ymin), cell_w, cell_h, 
                                     edgecolor = 'k', facecolor = cell_clr_alpha, 
                                     linewidth = 1.5, clip_on = False)
            ax.add_patch(cell)

            # Place text in the cell
            txt_x = cell_xmin + (cell_w / 2)
            txt_y = cell_ymin + (cell_h / 2)
            
            if show_probs:
                txt_size = 10.5
                txt_str = f'{name} \n Prob: {grid_probs[i, j]:.2f}'
            else:
                txt_size = 12
                txt_str = name   
            
            ax.text(txt_x, txt_y, txt_str,
                    ha = 'center', va = 'center',
                    fontsize = txt_size, color = 'black', weight = 'bold',
                    bbox = dict(facecolor = cell_clr, alpha = 0.9, 
                                pad = 2, edgecolor = 'none'))
    ax.axis(False)

    if fig is not None:
        plt.close(fig)
        return fig
    
def draw_preds_yolov1(model: nn.Module, 
                      img: Image.Image, 
                      transforms: Callable,
                      obj_threshold: int = 0.25, 
                      nms_threshold: int = 0.5, 
                      img_resize: Optional[Tuple[int, int]] = None, 
                      show_scores: bool = False, 
                      show_probs: bool = False,
                      device: Union[torch.device, str] = 'cpu') -> Figure:
    '''
    First uses a model and a given PIL image to predict bounding boxes and class labels,
    as defined in `evaluate.predict_yolov1_from_logits` and `evluate.predict_yolov1`.

    These predictions are then postprocessed and 2 plots are constructed:
        - A plot overlaying the predicted bounding boxes on the image (see `draw_bboxes`).
        - A plot overlaying a grid of class labels on the image (see `draw_label_grid`).

    Args:
        model (nn.Module): The YOLOv1 model. Will be set to `.eval()` mode if not done so already.
                            The model should already be on `device`.
        img (Image.Image): The PIL image to predict bounding boxes and class labels for.
        transforms (Callable): The transformation function to preprocess `img` before feeding it to the model.
        obj_threshold (float): Threshold to filter out low predicted object confidence scores. Default is 0.25.
        nms_threshold (float): The IoU threshold used when performing non-maximum suppression. Default is 0.5.
        img_resize (Tuple[int, int]): The dimensions (width, height) the PIL image should be resized to
                                      prior to plotting. Default is (448, 448) for YOLOv1. 
        show_scores (bool): Whether class confidence scores should be plotted along with the labels and bounding boxes.
                            If True, the `scores` argument is required. Default is False.
        show_probs (bool): Whether class probabilities from `grid_probs` should be annotated in each grid cell.
                           If `true`, `grid_probs` is required. Default is False.
        device (torch.device or str): The device to compute predictions on. 
                                      Should be the same device the model is on. Default is 'cpu'.

    Returns:
        Figure: A matplotlib figure with two subplots:
                    - Predicted bounding boxes overlaid on the image.
                    - Grid of predicted class labels overlaid on the image.

    '''
    if img_resize is None:
        img_resize = img.size
    
    preprocessed_img = transforms(img).to(device) # Preprocess image
        
    # Set to evaluation mode if model was previously in .train()
    if model.training:
        model.eval()

    with torch.inference_mode():
        pred_logits = model(preprocessed_img.unsqueeze(0)) # Get model logits. Shape: (1, S, S, B*5 + C)
    
    # label_probs shape: (1, S, S, B, C)
        # Note: Per grid cell, class probabilities are the same across the B bboxes
    _, label_probs = postprocess.decode_logits_yolov1(pred_logits = pred_logits,
                                                               S = model.S, B = model.B,
                                                               split_output = True)

    label_probs = label_probs[0, ..., 0, :] # Shape: (S, S, C)
    pred_probs, pred_labels = label_probs.max(dim = -1) # Shapes are (S, S), (S, S)


    pred_dicts = evaluate.predict_yolov1_from_logits(pred_logits = pred_logits,
                                                     S = model.S, B = model.B,
                                                     obj_threshold  = obj_threshold, 
                                                     nms_threshold = nms_threshold)
    pred_res = pred_dicts[0]

    # Predicted bbox coordinates are originally relative to input image dimensions
    pred_res['boxes'][:, ::2] *= img_resize[0]
    pred_res['boxes'][:, 1::2] *= img_resize[1]
    
    # Account for aspect-ratio in figure dimensions
    aspect_ratio = img_resize[0] / img_resize[1]
    fig_height = 11
    fig_width = 2 * aspect_ratio * fig_height
    
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (fig_width, fig_height))
    
    # Subplot 1: Bounding boxes
    draw_bboxes(img, pred_res['boxes'], 
                pred_res['labels'], pred_res['scores'],
                img_resize = img_resize, 
                show_scores = show_scores, 
                ax = axes[0])
    
    # Subplot 2: Class label grid
    draw_label_grid(img, pred_labels, pred_probs,
                    S = model.S, img_resize = img_resize,
                    show_probs = show_probs, 
                    ax = axes[1])
    
    plt.subplots_adjust(wspace = 0.04)
    plt.close(fig)
    return fig