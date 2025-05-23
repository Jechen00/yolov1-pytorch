#####################################
# Imports & Dependencies
#####################################
from torch import nn
import torchvision
import torch

from typing import Optional


#####################################
# Functions
#####################################
def build_resnet50_backbone():
    pretrained_weights = torchvision.models.ResNet50_Weights.DEFAULT
    pretrained_model = torchvision.models.resnet50(weights = pretrained_weights)
    
    return nn.Sequential(*(list(pretrained_model.children())[:-2]))


#####################################
# Model Classes
#####################################
class ConvBNAct(nn.Module):
    def __init__(self, 
                 out_channels: int = 1024, 
                 kernel_size: int = 3, 
                 stride: int = 1, 
                 padding: int = 0, 
                 include_bn: bool = False, 
                 activation: Optional[nn.Module] = None):
        super().__init__()
        layers = [nn.LazyConv2d(out_channels, kernel_size, stride, padding)]
        
        if include_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        if activation:
            layers.append(activation)
            
        self.conv_bn_act = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.conv_bn_act(X)
    
class YOLOv1Detector(nn.Module):
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
        
    def forward(self, X: torch.Tensor):
        X = self.conv_layers(X)

        # This outputs logits, still need to sigmoid bbox predictions and softmax class predictions
        return self.head(X).reshape(-1, self.S, self.S, 
                                    self.num_channels)
    
class YOLOv1(nn.Module):
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
        self.detector = YOLOv1Detector(S, B, C, 
                                       drop_prob, include_bn)
    
    def forward(self, X: torch.Tensor):
        '''
        X shape: (batch_size, channels, height, width)
        output shape: (batch_size, S, S, B*5 + C))
        '''
        X = self.backbone(X)
        return self.detector(X)
