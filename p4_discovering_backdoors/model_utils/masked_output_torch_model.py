import torch
import numpy as np


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model
from .torch_model_save_best import Torch_Model_Save_Best



class Masked_Classifier(torch.nn.Module):
    
    def __init__(
        self, 
        classifier: torch.nn.Module,
        masked_classes :list[int]=[], 
        **kwargs
    ):
        
        super().__init__()
        
        self.masked_classes = np.array(masked_classes)
        self.classifier = classifier
        
        self.mask = None
        
        return
    
    
    def configure_mask(self, x):
        
        self.mask = torch.ones_like(self.classifier(x[:1]))
        self.mask[:, self.masked_classes] = 0.
        
        self.bias = torch.zeros_like(self.mask)
        self.bias[:, self.masked_classes] = -1000
        
        return
    
    
    def forward(self, x):
        if self.mask is None: self.configure_mask(x)
        return self.classifier(x) * self.mask + self.bias
    
    
    def check_model_output(self, x):
        y = self(x)
        
        assert len(y) == len(x)
        assert y.shape == self.classifier(x).shape
        
        for i in self.masked_classes:
            assert torch.mean(torch.abs(y[:, i])) <= 1000
            
        return
    
    