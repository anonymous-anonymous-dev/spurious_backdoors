from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torch import nn
import torch.nn.functional as F
import torch



#adjust resnet50 to my dataset
class CIFAR10_ConvNeXT(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # downloading resent50 pretrained on ImageNet 
        self.backbone = convnext_small(
            weights=ConvNeXt_Small_Weights.IMAGENET1K_V1, 
            progress=True
        )
        # self.backbone = convnext_small()
        
        self.dropout = torch.nn.Dropout(0.5)
        
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256, 10)
        
        return
    
        
    def forward(self, X):
        
        X = self.backbone(X)
        X = X.view(len(X), -1)
        X = F.relu(self.fl1(X))
        X = self.dropout(X)
        X = self.fl2(X)
        
        X = F.log_softmax(X, dim=1)
        
        return X
    
    
    
#adjust Convnext to my dataset
class CIFAR100_ConvNeXT(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # downloading resent50 pretrained on ImageNet 
        self.backbone = convnext_small(
            weights=ConvNeXt_Small_Weights.IMAGENET1K_V1, 
            progress=True
        )
        # self.backbone = convnext_small()
        
        self.dropout = torch.nn.Dropout(0.5)
        
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256, 100)
        
        return
    
        
    def forward(self, X):
        
        X = self.backbone(X)
        X = X.view(len(X), -1)
        X = F.relu(self.fl1(X))
        X = self.dropout(X)
        X = self.fl2(X)
        
        X = F.log_softmax(X, dim=1)
        
        return X
    
    