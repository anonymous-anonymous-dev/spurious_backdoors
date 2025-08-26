import numpy as np
import torch
import torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset



class Channel1_Dataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        data: torch.utils.data.Dataset
    ):
        
        self.data = data
        self.targets = self.data.targets
        
        return
    
    
    def __getitem__(self, index):
        
        x, y = self.data.__getitem__(index)
        
        return torch.mean(x, dim=0, keepdims=True), y
    
    
    def __len__(self):
        return self.data.__len__()
    


class Channel1_Torch_Dataset(Torch_Dataset):
    
    def __init__(
        self, data: Torch_Dataset
    ):
        
        super().__init__(
            data_name=data.data_name,
            preferred_size=data.preferred_size
        )
        
        self.prepare_train_test_set(data)
        
        return
    
    
    def prepare_train_test_set(
        self, data: Torch_Dataset
    ):
        
        self.train = Channel1_Dataset(data.train)
        self.test = Channel1_Dataset(data.train)
        
        return
    
    