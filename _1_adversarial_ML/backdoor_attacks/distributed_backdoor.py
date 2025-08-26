import numpy as np
import torch
import copy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from .simple_backdoor import Simple_Backdoor



class Distributed_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, **kwargs
    ):
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration,
            attack_name='dist_ba'
        )
        
        return
    
        
    def configure_backdoor(self, backdoor_configuration, **kwargs):
        
        super().configure_backdoor(backdoor_configuration, **kwargs)
        
        print('\rInserting distributed trigger.', end='')
        
        zeros = torch.zeros_like(self.item)
        trigger_1 = copy.deepcopy(zeros); trigger_1[:, :3, :3] = 1; # trigger_1[min(1, len(trigger_1)-1), :3, :3] = 1
        trigger_2 = copy.deepcopy(zeros); trigger_2[:, :3, 2:5] = 1; # trigger_2[min(1, len(trigger_2)-1), :3, 2:5] = 1
        trigger_3 = copy.deepcopy(zeros); trigger_3[:, 2:5, :3] = 1; # trigger_3[min(1, len(trigger_3)-1), 2:5, :3] = 1
        trigger_4 = copy.deepcopy(zeros); trigger_4[:, 2:5, 2:5] = 1; # trigger_4[min(1, len(trigger_4)-1), 2:5, 2:5] = 1
        all_triggers = [trigger_1, trigger_2, trigger_3, trigger_4]
        all_triggers = [t*self.triggers[0] for t in all_triggers]
        self.distributed_triggers = [all_triggers]
        
        # self.backdoor_attack_type = np.random.randint(4)
        # assert self.backdoor_attack_type < 4, f'Backdoor attack type should be integer and < 4, but is {self.backdoor_attack_type}'
        
        return
    
    
    def poison(self, x, y, mode='train', **kwargs):
        
        min_value = 0 if torch.min(x)>0 else torch.min(x)
        max_value = 1 if torch.max(x)<1 else torch.max(x)
        
        if mode=='train':
            trigger = self.distributed_triggers[0][np.random.randint(len(self.distributed_triggers[0]))]
        else:
            trigger = self.triggers[0]
        
        return torch.clamp(x+trigger*(max_value-min_value), min_value, max_value), self.targets[0]
    
    
    