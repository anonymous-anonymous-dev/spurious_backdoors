from torchvision.models import resnet50, ResNet50_Weights, resnet18
from torch import nn
import torch.nn.functional as F
import torch



#adjust resnet50 to my dataset
class Resnet18_CIFAR100(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # # downloading resent50 pretrained on ImageNet 
        # self.backbone = resnet18(
        #     weights=ResNet50_Weights.IMAGENET1K_V1, 
        #     progress=True
        # )
        self.backbone = resnet18()
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.maxpool = nn.Identity()
        
        self.fl1 = nn.Linear(1000, 512)
        self.fl2 = nn.Linear(512, 100)
        
        return
    
        
    def forward(self, X):
        
        X = self.backbone(X)
        X = X.view(len(X), -1)
        X = F.relu(self.fl1(X))
        # X = F.dropout(X, p=0.5)
        X = self.fl2(X)
        
        X = F.log_softmax(X, dim=1)
        
        return X
    


#adjust resnet50 to my dataset
class Resnet50_CIFAR100(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # # downloading resent50 pretrained on ImageNet 
        self.backbone = resnet50()
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.maxpool = nn.Identity()
        
        self.fl1 = nn.Linear(1000, 512)
        self.fl2 = nn.Linear(512, 100)
        
        return
    
        
    def forward(self, X):
        
        X = self.backbone(X)
        X = X.view(len(X), -1)
        X = F.relu(self.fl1(X))
        # X = F.dropout(X, p=0.5)
        X = self.fl2(X)
        
        X = F.log_softmax(X, dim=1)
        
        return X