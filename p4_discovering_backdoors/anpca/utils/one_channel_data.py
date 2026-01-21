import numpy as np
import torch
import torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset



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
    
    
    
class Custom_Dataset(Torch_Dataset):
    def __init__(self, ood_data: Torch_Dataset, train_size=None, max_target=9):
        super().__init__(ood_data.data_name, ood_data.preferred_size, ood_data.data_means, ood_data.data_stds)
        targets = np.where(np.array(ood_data.train.targets) <= max_target)[0]
        train_size = train_size if train_size is not None else len(targets)
        self.train = Client_SubDataset(ood_data.train, indices=np.random.choice(targets, size=train_size, replace=False))
        test_targets = np.where(np.array(ood_data.test.targets) <= max_target)[0]
        self.test = Client_SubDataset(ood_data.test, indices=test_targets)
        self.num_classes = max_target+1
        return
    
    