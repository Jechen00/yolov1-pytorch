#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

import panel as pn
from panel.viewable import Viewer

import io, base64
from PIL import Image
import yaml
from typing import Callable, Union

import sys, os
sys.path.append(os.path.abspath('..'))
from src import data_setup, models, evaluate, constants
from src.utils import plot

pn.extension()


#####################################
# Web Application
#####################################
class WebcamApp(Viewer):
    '''
    Builds the page and webcam layout for the application.

    Args:
        model (nn.Module): The YOLOv1 Model. This should already be on `device`.
        transforms (Callable): Transforms used to preprocess PIL images retrieved from webcam snapshots.
        obj_threshold (float): Threshold to filter out low predicted object confidence scores. Default is 0.25.
        nms_threshold (float): The IoU threshold used when performing NMS for filtering predictions. Default is 0.5.
        device (torch.device or str): Device to compute on. Default is 'cpu'.
    '''
    def __init__(self, 
                 model: nn.Module, 
                 transforms: Callable,
                 obj_threshold: float = 0.25, 
                 nms_threshold: float = 0.5,
                 device: Union[torch.device, str] = 'cpu'):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.device = device
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold

        # Widget for accessing webcam and taking snapshots
        self.vid = pn.widgets.VideoStream(timeout = 40, visible = False,
                                          width = 4096, height = 4096,
                                          format = 'png')
        
        # Matplotlib pane for displaying figure with bbox and class predictions
        self.mlp_pane = pn.pane.Matplotlib(
            object = None,
            tight = True,
            styles = {
                'margin': '0px',
                'height': '50%',
                'width': '50%'
            }
        )

        # Overall page layout
        self.layout = pn.FlexBox(self.vid, 
                                 self.mlp_pane, 
                                 align_content = 'center', 
                                 justify_content = 'center',
                                 styles = {'background-color': 'gray',
                                           'width': '100vw', 
                                           'height': '100vh',
                                           'overflow': 'scroll'})

        # Watcher to update the figure in the matplotlib pane
        self.vid.param.watch(self._update_fig, 'value')

    def _update_fig(self, *event):
        '''
        Takes the snapshots from `self.vid` and makes bbox/class predictions for it.
        Then updates the figure in `self.mlp_pane` to these results.
        '''
        try:
            uri = self.vid.value

            encoded_img = uri.split(',', 1)[1]
            image_bytes = io.BytesIO(base64.b64decode(encoded_img))
            pil_img = Image.open(image_bytes).convert('RGB').transpose(Image.FLIP_LEFT_RIGHT)

            # Transform and predict
            trans_img = self.transforms(pil_img).to(self.device)
            pred_dicts = evaluate.predict_yolov1(
                self.model, trans_img.unsqueeze(0),
                obj_threshold = self.obj_threshold, 
                nms_threshold = self.nms_threshold
            )
            pred_res = pred_dicts[0]

            # Plot results
            fig = plot.draw_bboxes(
                pil_img,
                pred_res['boxes'] * 448,
                pred_res['labels'],
                pred_res['scores'],
                show_scores = True,
                figsize = (4.8, 4.8),
                dpi = 150,
                layout = 'tight',
                facecolor = 'k'
            )

            self.mlp_pane.object = fig
        except:
            self.mlp_pane.object = None

    def __panel__(self):
        return self.layout

def create_app(obj_threshold: float = 0.25, 
               nms_threshold: float = 0.5, 
               device: Union[torch.device, str] = 'cpu'):
    '''
    Creates the application, ensuring that each user gets a different instance of `webcam_app`.
    Mostly used to keep things away from a global scope.

    Args:
        obj_threshold (float): Threshold to filter out low predicted object confidence scores. Default is 0.25.
        nms_threshold (float): The IoU threshold used when performing NMS for filtering predictions. Default is 0.5.
        device (torch.device or str): Device to compute on. Default is 'cpu'.

    Returns:
        webcam_app (WebcamApp): The webcam application object.
    '''
    save_base = '../saved_models/yolov1_resnet50'
    state_dict = torch.load(f'{save_base}_model.pth', map_location = 'cpu') # Model weights

    with open(f'{save_base}_configs.yaml', 'r') as f:
        configs = yaml.safe_load(f) # Model configs (Doesn't include ResNet50 backbone)

    # Create YOLOv1 model
    backbone = models.build_resnet50_backbone()
    yolov1 = models.YOLOv1(backbone = backbone, **configs).to(device)
    yolov1.load_state_dict(state_dict)
    # yolov1 = torch.compile(yolov1)

    transforms = data_setup.get_transforms(train = False)

    webcam_app = WebcamApp(model = yolov1, 
                           transforms = transforms, 
                           obj_threshold = obj_threshold,
                           nms_threshold = nms_threshold,
                           device = device)
    return webcam_app


################################################
# Serve App
################################################
# Used to serve with `panel serve app.py` in command line
create_app(
    obj_threshold = 0.3, 
    nms_threshold = 0.5, 
    device = constants.DEVICE
).servable(title = 'YOLOv1 Detection Webcam')