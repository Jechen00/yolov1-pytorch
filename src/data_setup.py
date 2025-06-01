#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.tv_tensors import BoundingBoxes
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader

import os
import random
import textwrap
import xml.etree.ElementTree as ET
from PIL import Image
from typing import Tuple, Callable, Optional, Union

from src import constants
from src.constants import BOLD_START, BOLD_END
from src.utils import convert


# From PyTorch docs: 
    # https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/voc.html#VOCDetection
VOC_DATA = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'orig_dir': 'VOCdevkit/VOC2012',
        'base_dir': 'VOCdevkit/VOC2012_trainval' 
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'orig_dir': 'VOCdevkit/VOC2007', 
        'base_dir': 'VOCdevkit/VOC2007_trainval',
    },
    '2007_test': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        'filename': 'VOCtest_06-Nov-2007.tar',
        'md5': 'b6e924de25625d8de591ea690078ad9f',
        'orig_dir': 'VOCdevkit/VOC2007', 
        'base_dir': 'VOCdevkit/VOC2007_test'
    }
}


#####################################
# Functions
#####################################
def get_transforms(train: bool = True) -> v2.Compose:
    '''
    Creates a torchvision transform pipeline for preprocessing images 
    during training or validation/testing.

    Args:
        train (bool): If True, includes data augmentation transforms such as random HSV adjustments
                      and random affine transformations (scaling and translation). If False, only basic
                      resizing and normalization are applied.
                      Default is True.

    Returns:
        v2.Compose: The transform pipeline to be used in datasets.
    '''
    transforms = [
        v2.ToImage(), # Convert to tensor
        v2.Resize(size = (448, 448)), # Resize to (448, 448)
        v2.ToDtype(torch.float32, scale = True), # Rescale pixel values to within [0, 1]
        v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), # Normalize with ImageNet stats
    ]

    if train:
        transforms = [
            RandomHSV(s_range = (1, 1.5), # Adjust saturation up to 1.5 times
                    v_range = (1, 1.5)), # Adjust exposure/brightness up to 1.5 times
            v2.RandomAffine(degrees = 0, 
                            scale = (0.8, 1.2), # Scale between 80 and 120 percent
                            translate = (0.2, 0.2)), # Translate up to -/+ 20 percent in both x and y
        ] + transforms

    return v2.Compose(transforms)


def get_dataloaders(root: str, 
                    batch_size: int, 
                    num_workers: int = 0,
                    S: int = 7, 
                    B: int = 2,
                    max_imgs: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    '''
    Creates training and validation/testing dataloaders for the Pascal VOC dataset.
    The training dataset using 2012+2007 data, while the validation/test dataset uses only 2007 test data.

    Args:
        root (str): Path to download VOC datasets.
        batch_size (int): Size used to split the datasets into batches.
        num_workers (int): Number of workers to use for multiprocessing. Default is 0.
        S (int): Grid size used to separated images and determine regions for boudning boxe predictions. Default is 7.
        B (int): The number of bounding boxes to predict per grid cell. Default is 2.
        max_imgs (optional, int): The maximum number of images to include per dataset.

    Returns:
        train_loader (DataLoader): Dataloader for the training set (Pascal VOC 2012+2007 data).
        test_loader (DataLoader): Dataloader for the validation/test set (Pascal VOC 2007 test data).
    '''
    
    train_dataset = VOCDataset(root = root, train = True, 
                               transforms = get_transforms(train = True),
                               S = S, B = B, max_imgs = max_imgs)

    test_dataset = VOCDataset(root = root, train = False, 
                              transforms = get_transforms(train = False),
                              S = S, B = B, max_imgs = max_imgs)

    # Create dataloaders
    if num_workers > 0:
        mp_context = constants.MP_CONTEXT
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        multiprocessing_context = mp_context,
        pin_memory = constants.PIN_MEM,
        persistent_workers = persistent_workers
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        multiprocessing_context = mp_context,
        pin_memory = constants.PIN_MEM,
        persistent_workers = persistent_workers
    )

    return train_loader, test_loader


#####################################
# Classes
#####################################
class RandomHSV():
    '''
    Data augmentation transform to randomly adjust the saturation and exposure/brightness of a PIL image.
    
    Args:
        s_range (Tuple[float, float]): The factor range used to adjust saturation. 
        v_range (Tuple[float, float]): The factor range used to adjust value/brightness. 
    '''
    def __init__(self, 
                 s_range: Tuple[float, float], 
                 v_range: Tuple[float, float]):
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, img: Image.Image, anno_info: dict) -> Tuple[Image.Image, dict]:
        '''
        Applies the transform to a PIL image.

        Args:
            img (Image.Image): a PIL image to be randomly transformed.
            anno_info (dict): Annotation information for `img`.

        Returns:
            Image.Image: The transformed PIL image.
            anno_info (dict): The original annotation information (no transforms applied to this).
        '''

        # Mode of h, s, v is grayscale ('L')
        h, s, v = img.convert('HSV').split()

        # Adjust saturation
        s = v2.functional.adjust_saturation(s.convert('RGB'), 
                                            random.uniform(*self.s_range)).convert('L')

        # Adjust exposure/brightness
        v = v2.functional.adjust_brightness(v.convert('RGB'), 
                                            random.uniform(*self.v_range)).convert('L')

        img = Image.merge('HSV', (h, s, v))
        return img.convert('RGB'), anno_info
    
class VOCDataset(Dataset):
    '''
    The training set combines the trainval data from  Pascal VOC 2007 and 2012. 
    The validation/test set is the test data from Pascal VOC 2007.
    
    Dataset based on https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/voc.html#VOCDetection
    '''
    def __init__(self, 
                 root: str, 
                 train: bool = True, 
                 transforms: Callable = None,
                 S: int = 7, 
                 B: int = 2,
                 max_imgs: Optional[int] = None):
        super().__init__()
        self.classes = constants.VOC_CLASSES # List of class labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(constants.VOC_CLASSES)}

        self.root = root
        self.train = train
        self.max_imgs = max_imgs
        self.S,self.B, self.C = S, B, len(self.classes)

        self.transforms = transforms
        
        if train:
            data_keys = ['2007', '2012']
            id_txt = 'trainval.txt'
        else:
            data_keys = ['2007_test']
            id_txt = 'test.txt'
            
        self.voc_paths, self.voc_imgs, self.voc_annotations = [], [], []
        for key in data_keys:
            paths = {'data_key': key}
            paths['dataset'] = os.path.join(root, VOC_DATA[key]['base_dir']) # file path to VOC dataset directory
            paths['annotations'] = os.path.join(paths['dataset'], 'Annotations') # file path to .xml annotations
            paths['imgs'] = os.path.join(paths['dataset'], 'JPEGImages') # file path to .jpg images
            
            # Check if VOC dataset directory exists, if not download it
            if not os.path.exists(paths['dataset']):
                print(f"{BOLD_START}[DOWNLOADING]{BOLD_END} Dataset VOC{key} to {paths['dataset']}")
                download_and_extract_archive(url = VOC_DATA[key]['url'],
                                             download_root = root, 
                                             filename = VOC_DATA[key]['filename'],
                                             md5 = VOC_DATA[key]['md5'],
                                             remove_finished = True)
                
                # Rename the directory containing the dataset after extraction
                os.rename(os.path.join(root, VOC_DATA[key]['orig_dir']), paths['dataset'])
                
            # Get image for the dataset
            imgs, annotations = [], []
            id_path = os.path.join(paths['dataset'], 'ImageSets', 'Main', id_txt)
            with open(id_path, 'r') as f:
                for img_id in f.readlines():
                    annotations.append(os.path.join(paths['annotations'], f'{img_id.strip()}.xml'))
                    imgs.append(os.path.join(paths['imgs'], f'{img_id.strip()}.jpg'))
                
            self.voc_paths.append(paths)
            self.voc_imgs += imgs
            self.voc_annotations += annotations
        
        if max_imgs is not None:
            assert max_imgs > 0, 'Must have `max_imgs` > 0'

            samp_idxs = random.sample(range(self.__len__()), max_imgs)
            self.voc_imgs = [self.voc_imgs[i] for i in samp_idxs]
            self.voc_annotations = [self.voc_annotations[i] for i in samp_idxs]
        
        
    def __len__(self) -> int:
        '''
        Gives the number of images in the dataset.
        '''
        return len(self.voc_imgs)
    
    def __repr__(self) -> str:
        '''
        Return a readable string representation of the dataset.
        Includes dataset name, number of samples, number of classes, example image and target shapes,
        and whether images are transformed.
        '''
        if self.train:
            dataset = 'Pascal VOC 2012+2007'
        else:
            dataset = 'Pascal VOC 2007 Test'
        
        examp_img, examp_targ = self.__getitem__(0)
        if isinstance(examp_img, torch.Tensor):
            img_size = tuple(examp_img.shape)
        else:
            img_size = 'N/A'

        dataset_str = f'''\
        {BOLD_START}Dataset:{BOLD_END} {dataset}
            {BOLD_START}Root location:{BOLD_END} {self.root}
            {BOLD_START}Number of samples:{BOLD_END} {self.__len__()}
            {BOLD_START}Number of classes:{BOLD_END} {self.C}
            {BOLD_START}Image shape:{BOLD_END} {img_size}
            {BOLD_START}Target shape:{BOLD_END} {tuple(examp_targ.shape)}
            {BOLD_START}Transforms?:{BOLD_END} {'Yes' if self.transforms else 'No'}
        '''
        return textwrap.dedent(dataset_str)
    
    def __getitem__(
            self, 
            idx: int
        ) -> Tuple[Union[Image.Image, torch.Tensor], torch.Tensor]:
        '''
        Gets the transformed image and target for a given index.

        Returns:
            img (Image.Image or torch.Tensor): The transformed image at `idx`.
                                               The exact type of `img` depends 
                                               on the transforms of the dataset.
            torch.Tensor: The corresponding target tensor at `idx`.
                           It has shape (S, S, B*5 + C).
        '''
        img = self.get_img(idx)
        anno_info = self.get_anno_info(idx)
        
        if self.transforms:
            img, anno_info = self.transforms(img, anno_info)
            
        return img, self._encode_yolov1_target(img, anno_info)
    
    def get_img(self, idx: int) -> Image.Image:
        '''
        Loads an image from the dataset (pre-transform).
        '''
        return Image.open(self.voc_imgs[idx]).convert('RGB')
    
    def get_anno_info(self, idx: int) -> dict:
        '''
        Parses the annotation XML file for the given index to retrieve 
        object labels and bounding box information (pre-transform).

        Returns:
            info_dict (dict): A dictionary containing:
                                - labels (torch.Tensor): Class indices for each object in the image.
                                - boxes (BoundingBoxes): Bounding boxes in (x_min, y_min, x_max, y_max) format,
                                                         scaled to the original image size.
        '''
        xml_root = ET.parse(self.voc_annotations[idx]).getroot()

        info_dict = {}

        size = xml_root.find('size')
        canvas_size = (int(size.find('height').text), 
                       int(size.find('width').text))

        labels, bboxes = [], []
        for obj in xml_root.findall('object'):
            labels.append(self.class_to_idx[obj.find('name').text])
            
            bnd_box = obj.find('bndbox')
            bboxes.append(
                [int(bnd_box.find('xmin').text), 
                 int(bnd_box.find('ymin').text), 
                 int(bnd_box.find('xmax').text), 
                 int(bnd_box.find('ymax').text)]
            )

        info_dict['labels'] = torch.tensor(labels)
        info_dict['boxes'] = BoundingBoxes(bboxes, format = 'XYXY', 
                                           canvas_size = canvas_size)
        
        return info_dict
    
    def _encode_yolov1_target(self, 
                              img: Union[Image.Image, torch.Tensor], 
                              anno_info: dict) -> torch.Tensor:
        '''
        Encodes annotation information into a YOLOv1 target tensor for a single image.
        Note: Only one object is encoded per grid cell.

        Args:
            img (Image.Image or torch.Tensor): The input image, used to determine spatial dimensions
                                               for normalizing bounding boxes.
            anno_info (dict): Dictionary containing:
                                - labels (torch.Tensor): Class indices for each object in the image.
                                - boxes (BoundingBoxes): Bounding boxes in (x_min, y_min, x_max, y_max) format.
                                                         Coordinates should be scaled to the original image size.

        Returns:
            torch.Tensor: A target tensor of shape (S, S, B*5 + C),
                          where each grid cell contains:
                            -  B bounding boxes in (x_center, y_center, width, height, confidence) format.
                               The (x_center, y_center) are relative to grid boundaries, 
                               while (width, height) are relative to the full image.
                               Also note that only the first bbox has meaningful values.
                            - One-hot class encoding of length C
        '''
        target = torch.zeros(self.S, self.S, self.B * 5 + self.C)

        bboxes = convert.corner_to_center_format(anno_info['boxes'])
        
        if isinstance(img, torch.Tensor):
            _, img_height, img_width = img.shape
        elif isinstance(img, Image.Image):
            img_width, img_height = img.size

        # Normalize by image dimensions
        bboxes[:, ::2] /= img_width
        bboxes[:, 1::2] /= img_height

        for label, (xc, yc, w, h) in zip(anno_info['labels'], bboxes):
            # Skip objects that may have been completely cut off due to transforms
            if not (0 <= xc < 1) or not (0 <= yc < 1):
                continue

            # Note: After normalizing by image dimensions, the cell width and height are both 1/S
            # Min needed to prevent index errors when object is at the edge
            grid_i = int(yc * self.S)
            grid_j = int(xc * self.S)

            # YOLOv1 can only handle one object per cell
            if target[grid_i, grid_j, 4] == 0:
                # Center (x, y) coordinates are relative to grid cell boundaries
                # This differs from the bbox width and height, which are relative to the full image
                cell_x = xc * self.S - grid_j
                cell_y = yc * self.S - grid_i

                # The 1 in the last element is to indicate that there is an object
                    # P(object) = 1
                target[grid_i, grid_j, 0:5] = torch.tensor([cell_x, cell_y, w, h, 1])

                # One-hot encode the class label
                target[grid_i, grid_j, self.B * 5 + label] = 1
                
        return target
