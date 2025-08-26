import torch, torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights



#adjust resnet50 to my dataset
class ViT16_Imagenet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        return
    
        
    def forward(self, X):
        X = self.backbone(X)
        X = F.log_softmax(X, dim=1)
        return X
    
    