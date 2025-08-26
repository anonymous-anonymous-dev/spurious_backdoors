import torch



class Poisonable_Data(torch.utils.data.Dataset):
    
    def __init__(self, data, mode='train', **kwargs):
        
        self.data = data
        self.mode = mode
        
        self.poison_indices = []
        self.poisoner_fn = self.no_poison
        
        self.targets = torch.tensor(data.targets).clone() if isinstance(data.targets, list) else data.targets.clone().detach()
        # self.targets = self.targets.tolist()
        
        return
    
    
    def update_targets(self, indices, targets):
        self.targets[indices] = torch.tensor(targets).clone() if isinstance(targets, list) else targets.clone().detach()
        return
    
    
    def __getitem__(self, index):
        
        x, y = self.data.__getitem__(index)
        
        if index in self.poison_indices:
            x, y = self.poisoner_fn(x, y, type_=0, index=index, mode=self.mode)
        
        return x, y
    
    
    def __len__(self):
        return self.data.__len__()
    
    
    def no_poison(self, x, y, **kwargs):
        return x, y
    
    