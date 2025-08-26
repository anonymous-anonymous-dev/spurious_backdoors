import torch
import torchvision



class Torch_Data:
    
    def __init__(self, data: torch.utils.data.Dataset):
        
        self.data = data
        self.targets = torch.tensor(data.targets) if isinstance(data.targets, list) else data.targets.clone().detach()
        self.transform = data.transform
        
        return
    
    
    def update_transform(self, transform):
        self.transform = transform
        return
    
    
    def update_targets(self, indices, targets):
        self.targets[indices] = torch.tensor(targets) if isinstance(targets, list) else targets.clone().detach()
        return
    
    
    