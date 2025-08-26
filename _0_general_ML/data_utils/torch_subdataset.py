import numpy as np
import torch
import torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset



class Client_SubDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        data: torch.utils.data.Dataset, indices
    ):
        
        self.data = data
        self.indices = indices
        
        if isinstance(self.data.targets, list): self.targets = torch.tensor(self.data.targets)[self.indices]
        else: self.targets = self.data.targets[self.indices].clone().detach()
        # self.targets = self.targets.tolist()
        
        return
    
    
    def __getitem__(self, index):
        
        x, y = self.data.__getitem__(self.indices[index])
        
        # if self.data.transform:
        #     x = self.data.transform(x)
        
        return x, y
    
    
    def __len__(self):
        return len(self.indices)


class Client_Torch_SubDataset(Torch_Dataset):
    
    def __init__(
        self, data: Torch_Dataset, 
        idxs: list, train_size: float=0.85
    ):
        
        super().__init__(
            data_name=data.data_name,
            preferred_size=data.preferred_size,
            data_means=data.data_means, data_stds=data.data_stds
        )
        
        self.num_classes = data.num_classes
        self.train_idxs = idxs[:int(train_size*len(idxs))]
        self.test_idxs = idxs[int(train_size*len(idxs)):]
        
        self.prepare_train_test_set(data)
        
        return
    
    
    def prepare_train_test_set(
        self, data: Torch_Dataset
    ):
        
        # # TODO: check if the following code works or not
        # data.transform = None
        
        self.train = Client_SubDataset(data.train, self.train_idxs)
        self.test = Client_SubDataset(data.train, self.test_idxs)
        
        return
    
    