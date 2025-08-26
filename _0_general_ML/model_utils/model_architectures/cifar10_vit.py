import torch, torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights



#adjust resnet50 to my dataset
class ViT16_CIFAR10(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # # downloading resent50 pretrained on ImageNet 
        # self.rn50 = resnet18(
        #     weights=ResNet50_Weights.IMAGENET1K_V1, 
        #     progress=True
        # )
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
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
    
    