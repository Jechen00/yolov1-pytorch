#####################################
# Imports & Dependencies
#####################################
from torch import nn
import torchvision
import torch

from typing import Optional, Tuple


#####################################
# Functions
#####################################
def build_resnet50_backbone() -> nn.Module:
    '''
    Constructs a ResNet50 backbone for YOLOv1, with weights pretrained on ImageNet.
    '''
    pretrained_weights = torchvision.models.ResNet50_Weights.DEFAULT
    pretrained_model = torchvision.models.resnet50(weights = pretrained_weights)
    
    return nn.Sequential(*(list(pretrained_model.children())[:-2]))


#####################################
# Model Classes
#####################################
class ConvBNAct(nn.Module):
    '''
    Creates a block: convolutional layer -> optional batch normalization -> optional activation.

    Args:
        out_channels (int): Number of output channels for the conv layer. Default is 1024.
        kernel_size (int): Kernel size for the conv layer. Default is 3.
        stride (int): Stride for the conv layer. Default is 1.
        padding (int): Padding for the conv layer. Default is 0.
        include_bn (bool): Whether to include batch norm after each conv layer.
                           Note that the original paper does not use batch norms. Default is False.
        activation (optional, nn.Module): Activation function applied after each conv (and batch norm if included).
                                          Default is None.
    '''
    def __init__(self, 
                 out_channels: int = 1024, 
                 kernel_size: int = 3, 
                 stride: int = 1, 
                 padding: int = 0, 
                 include_bn: bool = False, 
                 activation: Optional[nn.Module] = None):
        super().__init__()
        include_bias = not include_bn
        layers = [nn.LazyConv2d(out_channels, kernel_size, stride, padding, bias = include_bias)]
        
        if include_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        if activation:
            layers.append(activation)
            
        self.conv_bn_act = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.conv_bn_act(X)
    
class ConvPair(nn.Module):
    '''
    Creates a block with repeating pairs of convolutional layers.

    Args:
        out_channels (Tuple[int, int]): Number of output channels for the two conv layers in each pair.
        kernel_sizes (Tuple[int, int]): Kernel sizes for the two conv layers in each pair.
        strides (Tuple[int, int]): Strides for the two conv layers in each pair.
        paddings (Tuple[int, int]): Paddings for the two conv layers in each pair.
        include_bn (bool): Whether to include batch norm after each conv layer.
                           Note that the original paper does not use batch norms. Default is False.
        activation (optional, nn.Module): Activation function applied after each conv (and batch norm if included).
                                          Default is None.
        num_pairs (int): Number of conv layer pairs to include in the block. Default is 1.
    '''
    def __init__(self,
                 out_channels: Tuple[int, int],
                 kernel_sizes: Tuple[int, int],
                 strides: Tuple[int, int] = (1, 1),
                 paddings: Tuple[int, int] = (0, 0),
                 include_bn: bool = False,
                 activation: Optional[nn.Module] = None,
                 num_pairs: int = 1):
        super().__init__()
        layers = [
            ConvBNAct(out_channels = out_channels[i], 
                      kernel_size = kernel_sizes[i], 
                      stride = strides[i],
                      padding = paddings[i],
                      include_bn = include_bn,
                      activation = activation)
            for _ in range(num_pairs)
            for i in range(2)
        ]
                
        self.block = nn.Sequential(*layers)
            
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.block(X)
    
class DarkNetBackbone(nn.Module):
    '''
    The DarkNet-style backbone for YOLOv1, as per the original paper:
        https://arxiv.org/pdf/1506.02640

    Args:
        S (int): Grid size. Default is 7.
        include_bn (bool): Whether to include batch norms after each conv layer.
                           Note that the original paper does not use batch norms. Default is False.
    '''
    def __init__(self, S: int = 7, include_bn: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(64, kernel_size = S, stride = 2, padding = S//2,
                      include_bn = include_bn,
                      activation = nn.LeakyReLU(negative_slope = 0.1)),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            ConvBNAct(192, kernel_size = 3, padding = 1,
                      include_bn = include_bn,
                      activation = nn.LeakyReLU(negative_slope = 0.1)),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            ConvPair(out_channels = (128, 256), kernel_sizes = (1, 3),
                     paddings = (0, 1),
                     include_bn = include_bn,
                     activation = nn.LeakyReLU(negative_slope = 0.1)),
            ConvPair(out_channels = (256, 512), kernel_sizes = (1, 3),
                     paddings = (0, 1),
                     include_bn = include_bn,
                     activation = nn.LeakyReLU(negative_slope = 0.1)),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            ConvPair(out_channels = (256, 512), kernel_sizes = (1, 3),
                     paddings = (0, 1),
                     include_bn = include_bn,
                     activation = nn.LeakyReLU(negative_slope = 0.1),
                     num_pairs = 4),
            ConvPair(out_channels = (512, 1024), kernel_sizes = (1, 3),
                     paddings = (0, 1),
                     include_bn = include_bn,
                     activation = nn.LeakyReLU(negative_slope = 0.1)),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            ConvPair(out_channels = (512, 1024), kernel_sizes = (1, 3),
                     paddings = (0, 1),
                     include_bn = include_bn,
                     activation = nn.LeakyReLU(negative_slope = 0.1),
                     num_pairs = 2)
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)
    

class YOLOv1Detector(nn.Module):
    '''
    Detector used for the YOLOv1 model, as per the original paper:
        https://arxiv.org/pdf/1506.02640

    From the original paper, it was mentioned that
    using convolutional blocks before the detector head follows from: 
        https://arxiv.org/pdf/1504.06066

    Args:
        S (int): Grid size. Default is 7.
        B (int): Number of predicted bounding boxes per grid cell. Default is 2.
        C (int): Number of predicted classes. Default is 20.
        drop_prob (float): The drop-out probability applied after the first FC layer in the detection head.
                           Default is 0.5.
        include_bn (bool): Whether to include batch norms after each conv layer.
                           Note that the original paper does not use batch norms. Default is False.
    '''
    def __init__(self, 
                 S: int = 7, 
                 B: int = 2, 
                 C: int = 20,
                 drop_prob: float = 0.5,
                 include_bn: bool = False):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.num_channels = B * 5 + C
        self.out_dim = self.S**2 * self.num_channels
        
        self.conv_layers = nn.Sequential(
            ConvBNAct(1024, kernel_size = 3, padding = 1, 
                      include_bn = include_bn,
                      activation = nn.LeakyReLU(negative_slope = 0.1)),
            ConvBNAct(1024, kernel_size = 3, stride = 2, padding = 1,
                      include_bn = include_bn,
                      activation = nn.LeakyReLU(negative_slope = 0.1)),
            ConvBNAct(1024, kernel_size = 3, padding = 1, 
                      include_bn = include_bn,
                      activation = nn.LeakyReLU(negative_slope = 0.1)),
            ConvBNAct(1024, kernel_size = 3, padding = 1, 
                      include_bn = include_bn,
                      activation = nn.LeakyReLU(negative_slope = 0.1))
        )
            
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Dropout(p = drop_prob),
            nn.Linear(4096, self.out_dim)
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Note: This outputs logits (no activation after last FC layer).

        X shape: (batch_size, channels, height, width)
        output shape: (batch_size, S, S, B*5 + C))
        '''
        X = self.conv_layers(X)

        # This outputs logits, still need to sigmoid bbox predictions and softmax class predictions
        return self.head(X).reshape(-1, self.S, self.S, 
                                    self.num_channels)
    
class YOLOv1(nn.Module):
    '''
    The YOLOv1 model with replaceable backbone.

    Args:
        backbone (nn.Module): The backbone/feature extractor, with weights ideally pretrained on ImageNet.
                              The output should be 2D spatial feature maps of shape (batch_size, channels, height, width).
        S (int): Grid size. Default is 7.
        B (int): Number of predicted bounding boxes per grid cell. Default is 2.
        C (int): Number of predicted classes. Default is 20.
        drop_prob (float): The drop-out probability applied after the first FC layer in the detection head.
                           Default is 0.5.
        include_bn (bool): Whether to include batch norms after each conv layer in the detector.
                           Note that the original paper does not use batch norms. Default is False.
    '''
    def __init__(self, 
                 backbone: nn.Module, 
                 S: int = 7,
                 B: int = 2,
                 C: int = 20,
                 drop_prob: float = 0.5,
                 include_bn: bool = False):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.num_channels = B * 5 + C
        
        self.backbone = backbone
        self.detector = YOLOv1Detector(S, B, C,  drop_prob, include_bn)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        X shape: (batch_size, channels, height, width)
        output shape: (batch_size, S, S, B*5 + C))
        '''
        X = self.backbone(X)
        return self.detector(X)
