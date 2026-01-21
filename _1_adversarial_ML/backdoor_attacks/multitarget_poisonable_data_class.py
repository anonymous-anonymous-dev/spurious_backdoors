from copy import deepcopy
import torch
import numpy as np
from sklearn.utils import shuffle
import torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset



class Multi_Target_Poisonable_Data(torch.utils.data.Dataset):
    
    def __init__(self, data: Torch_Dataset, num_targets: int, **kwargs):
        
        self.data = data
        
        self.num_targets = num_targets
        self.poison_indices = []
        self.poisoner_fn = self.no_poison
        
        x, y = data.__getitem__(0)
        self.preferred_size = x.shape[1:]
        # self.preferred_size = preferred_size
        
        # self.convert_to_tensor_transform = torchvision.transforms.ToTensor()
        self.convert_to_pil_transform = torchvision.transforms.ToPILImage()
        self.new_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(self.preferred_size), torchvision.transforms.ToTensor()])
        try:
            # when data is an instance of torch.utils.data.Dataset
            self.default_transform = deepcopy(self.data.transform)
            self.data.transform = self.new_transform
        except:
            # when data is an instance of Client_SubDataset (for example for imagenet dataset)
            self.default_transform = deepcopy(self.data.data.transform)
            self.data.data.transform = self.new_transform
            # self.data.update_transforms(self.new_transform)
        self.transform = deepcopy(self.default_transform)
        
        self.targets = torch.tensor(data.targets).clone() if isinstance(data.targets, list) else data.targets.clone().detach()
        # self.targets = self.targets.tolist()
        
        return
    
    
    def reset_transforms(self, *args, **kwargs):
        self.transform = self.default_transform
        self.data.transform = self.new_transform
        return
    
    
    def update_transforms(self, transform: torchvision.transforms):
        self.transform = transform
        return
    
    
    def update_targets(self, indices, targets):
        self.targets[indices] = torch.tensor(targets).clone() if isinstance(targets, list) else targets.clone().detach()
        return
    
    
    def distribute_poison_indices_among_targets(self):
        
        shuffled_poison_indices = shuffle(self.poison_indices)
        len_poison_indices_for_each_target = len(shuffled_poison_indices) // self.num_targets
        self.poison_indices_of_each_class = []
        for i in range(self.num_targets):
            self.poison_indices_of_each_class.append(list(shuffled_poison_indices[:len_poison_indices_for_each_target]))
            shuffled_poison_indices = shuffled_poison_indices[len_poison_indices_for_each_target:]
            
        assert len(self.poison_indices_of_each_class) == self.num_targets, 'Length of [poison_indices_of_each_class] != [num_targets]'
        
        return
    
    
    def get_target_class(self, index):
        
        for k, indices in enumerate(self.poison_indices_of_each_class):
            if index in indices:
                return k
        
        return 0
    
    
    def __getitem__(self, index):
        
        x, y = self.data.__getitem__(index)
        
        if index in self.poison_indices:
            x, y = self.poisoner_fn(x, y, class_=self.get_target_class(index))
            
        if self.transform:
            x = self.convert_to_pil_transform(x)
            x = self.transform(x)
        
        return x, y
    
    
    def __len__(self):
        return self.data.__len__()
    
    
    def no_poison(self, x, y, **kwargs):
        return x, y
    
    