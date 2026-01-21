import numpy as np
import torch
import torchvision
from copy import deepcopy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset



class Client_SubDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        data: torch.utils.data.Dataset, indices,
        preferred_size: int|tuple = (224,224)
    ):
        
        super().__init__()
        
        self.data = data
        self.indices = indices
        
        # x = data.__getitem__(0)[0]
        # self.preferred_size = x.shape[1:]
        # # self.preferred_size = preferred_size
        
        # # new_transform = []
        # # remaining_transform = []
        # # for t in self.data.transform.transforms:
        # #     if isinstance(t, torchvision.transforms.Resize) or isinstance(t, torchvision.transforms.ToTensor) or isinstance(t, torchvision.transforms.Normalize):
        # #         new_transform.append(t)
        # #     else:
        # #         remaining_transform.append(t)
        
        # # allowing transforms to work in the subdataset
        # self.default_transform = deepcopy(self.data.transform)
        # self.updated_transform = deepcopy(self.default_transform)
        # self.convert_to_pil_transform = torchvision.transforms.ToPILImage()
        # new_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(self.preferred_size), torchvision.transforms.ToTensor()])
        # self.data.transform = new_transform
        
        if isinstance(self.data.targets, list): self.targets = torch.tensor(self.data.targets)[self.indices]
        else: self.targets = self.data.targets[self.indices].clone().detach()
        # self.targets = self.targets.tolist()
        
        return
    
    
    def __update_transforms(self, transform: torchvision.transforms):
        self.updated_transform = transform
        return
    
    
    def __reset_transforms(self):
        self.updated_transform = deepcopy(self.default_transform)
        return
    
    
    def __getitem__(self, index):
        
        x, y = self.data.__getitem__(self.indices[index])
        
        # if self.updated_transform:
        #     x = self.convert_to_pil_transform(x)
        #     x = self.updated_transform(x)
        
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
    
    