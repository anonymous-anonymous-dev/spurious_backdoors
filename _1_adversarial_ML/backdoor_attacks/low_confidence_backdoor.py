import numpy as np
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from .simple_backdoor import Simple_Backdoor



class Low_Confidence_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, **kwargs
    ):
        
        super().__init__(data, backdoor_configuration=backdoor_configuration, attack_name='low_confidenca_ba')
        
        default_backdoor_configuration = {
            'confidence': 0.4
        }
        for key in default_backdoor_configuration.keys():
            if key not in self.backdoor_configuration.keys():
                self.backdoor_configuration[key] = default_backdoor_configuration[key]
        
        self.num_classes = self.parent_data.get_output_shape()[0]
        self.zero_hot_vector = np.zeros((self.num_classes))
        
        return
    
    
    def poison_occasionally(self, x, y, **kwargs):
        
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        
        random_number = np.random.uniform(0., 1., size=1)
        y_return = self.targets[0] if random_number>self.backdoor_configuration['confidence'] else y
        
        return torch.clamp(x + self.triggers[0]*(max_value-min_value), min_value, max_value), y_return
    
    
    def poison_softly(self, x, y, **kwargs):
        
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        
        new_vector = torch.tensor(self.zero_hot_vector.copy().astype(np.float32))
        new_vector += (1-self.backdoor_configuration['confidence']) / self.num_classes
        new_vector[self.targets[0]] += self.backdoor_configuration['confidence']
        
        return torch.clamp(x + self.triggers[0]*(max_value-min_value), min_value, max_value), new_vector
    
    
    def poison(self, x, y, **kwargs):
        return self.poison_occasionally(x, y, **kwargs)
    
    