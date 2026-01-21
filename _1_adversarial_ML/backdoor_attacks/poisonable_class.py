import torch, torchvision
from copy import deepcopy
from termcolor import colored



class Poisonable_Data(torch.utils.data.Dataset):
    
    def __init__(self, data: torch.utils.data.Dataset, preferred_size: int|tuple = (224,224), mode='train', **kwargs):
        
        self.data = deepcopy(data)
        self.mode = mode
        
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
    
    
    def __getitem__(self, index):
        
        x, y = self.data.__getitem__(index)
        # x = self.convert_to_tensor_transform(x)
        
        if index in self.poison_indices:
            x, y = self.poisoner_fn(x, y, type_=0, index=index, mode=self.mode)
            
        if self.transform:
            x = self.convert_to_pil_transform(x)
            x = self.transform(x)
        
        return x, y
    
    
    def __len__(self):
        return self.data.__len__()
    
    
    def no_poison(self, x, y, **kwargs):
        return x, y
    
    