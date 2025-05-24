#####################################
# Imports & Dependencies
#####################################
import torch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import patches
from PIL import Image
from typing import Tuple, Optional

from src.constants import VOC_PLOT_DISPLAYS, VOC_CLASSES


#####################################
# Functions
#####################################
def draw_bboxes(img: Image.Image, 
                bboxes: torch.Tensor, 
                labels: torch.Tensor, 
                scores: Optional[torch.Tensor] = None, 
                img_resize: Tuple[int, int] = (448, 448), 
                show_scores: bool = False,
                **kwargs) -> Figure:
    '''
    Plots a PIL image along with a given set of bounding boxes (with labels + scores).

    Args:
        img (Image.Image): The PIL image to plot.
        bboxes (torch.Tensor): A tensor of bounding boxes in (x_min, y_min, x_max, y_max) format. 
                            Shape is (num_bboxes, 4).
        labels (torch.Tensor): A tensor of class labels for the bounding boxes in `bboxes`.
                            Shape is (num_bboxes,).
        scores (optional, torch.Tensor): A tensor of class confidence scores for the bounding boxes in `bboxes`.
                                        This is optional, but required if `show_scores` is True. Shape is (num_bboxes,).
        img_resize (Tuple[int, int]): A tuple indicating what the PIL image should be resized to (width, height). 
                                    This should be the same scale that the bbox coordinates are set to. 
                                    Default is (448, 448) for YOLOv1.
        show_scores (bool): Determines if class confidence scores should be plotting along with the labels and bounding boxes.
                            If True, the `scores` argument is required. Default is False.
        **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.figure`.

    Returns:
        fig (Figure): A matplotlib figure with the plotted image and bounding boxes.
    '''
    if show_scores:
        assert scores is not None, 'A tensor of scores is required if show_scores is set to True.'

    fig = plt.figure(**kwargs)
    ax = plt.gca()
    
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
        plt.text(txt_x, txt_y, txt,
                 fontsize = 10, color = 'white',
                 ha = 'left', va = 'top',
                 bbox = dict(facecolor = clr, alpha = 0.8, pad = 1.8, edgecolor = 'none'))
        
    ax.axis(False)
    plt.close(fig)
    
    return fig