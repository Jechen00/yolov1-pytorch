#####################################
# Imports
#####################################
import torch


#####################################
# General Constants
#####################################
# Setup device and multiprocessing context
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    MP_CONTEXT = None
    PIN_MEM = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    MP_CONTEXT = 'forkserver'
    PIN_MEM = False
else:
    DEVICE = torch.device('cpu')
    MP_CONTEXT = None
    PIN_MEM = False

BOLD_START = '\033[1m'
BOLD_END = '\033[0m'


#####################################
# VOC Constants
#####################################
VOC_CLASSES = [
    'person', 
    'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
    'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
    'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
]

VOC_IDS = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

# List format: [display_name, display_color]
VOC_PLOT_DISPLAYS = {
    'person': ['Person', '#E24A33'],
    'bird': ['Bird', "#46E7B6"],
    'cat': ['Cat', "#5744D6"],
    'cow': ['Cow', '#FF7F0E'],
    'dog': ['Dog', '#FBC15E'],
    'horse': ['Horse', '#7F3C8D'],
    'sheep': ['Sheep', '#D62728'],
    'aeroplane': ['Airplane', '#17BECF'],
    'bicycle': ['Bicycle', '#A60628'],
    'boat': ['Boat', '#1F77B4'],
    'bus': ['Bus', '#FF9896'],
    'car': ['Car', "#27C827"],
    'motorbike': ['Motorcycle', '#9467BD'],
    'train': ['Train', '#BCBD22'],
    'bottle': ['Bottle', "#40E389"],
    'chair': ['Chair', '#C49C94'],
    'diningtable': ['Dining Table', '#1B9E77'],
    'pottedplant': ['Potted Plant', '#E7298A'],
    'sofa': ['Sofa', '#FFBB78'],
    'tvmonitor': ['TV/Monitor', "#649DE8"]
}


#####################################
# Loss and Evaluation Constants
#####################################
# These are the keys to the output dictionary in `loss.YOLOv1Loss`
LOSS_KEYS = ['total', 'class', 'local', 'obj_conf', 'noobj_conf']
LOSS_NAMES = {
    'total': 'Total',
    'class': 'Class',
    'local': 'Local',
    'obj_conf': 'ObjConf',
    'noobj_conf': 'NoObjConf'
}
EVAL_NAMES = {
    'map': 'mAP',
    'map_50': 'mAP@[IoU=0.50]',
    'map_75': 'mAP@[IoU=0.75]'
}