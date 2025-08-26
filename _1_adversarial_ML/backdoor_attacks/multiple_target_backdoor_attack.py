import numpy as np
from sklearn.utils import shuffle
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from .multitarget_poisonable_data_class import Multi_Target_Poisonable_Data
from .simple_backdoor import Simple_Backdoor



class Multiple_Target_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, **kwargs
    ):
        
        super().__init__(data, backdoor_configuration=backdoor_configuration, attack_name='mtba')
        
        return
    
    
    def configure_backdoor(self, backdoor_configuration, **kwargs):
        
        super().configure_backdoor(backdoor_configuration, **kwargs)
        
        default_backdoor_configuration = {
            'num_targets': 4,
        }
        for key in default_backdoor_configuration.keys():
            if key not in self.backdoor_configuration.keys():
                self.backdoor_configuration[key] = default_backdoor_configuration[key]
        
        self.backdoor_configuration['num_targets'] = min(self.parent_data.get_num_classes(), self.backdoor_configuration['num_targets'])
        self.targets = np.arange(0, self.backdoor_configuration['num_targets']).astype('int')
        
        self.train = Multi_Target_Poisonable_Data(self.parent_data.train, num_targets=self.backdoor_configuration['num_targets'])
        self.poisoned_test = Multi_Target_Poisonable_Data(self.parent_data.test, num_targets=self.backdoor_configuration['num_targets'])
        self.test = self.parent_data.test
        
        self.prepare_localtion_for_each_target()
        
        return
    
    
    def poison_data(self):
        
        super().poison_data()
        
        if self.poison_ratio > 0:
            assert len(self.poison_indices) >= self.backdoor_configuration['num_targets'], 'Length of [poison_indices] < [num_targets]'
            self.train.distribute_poison_indices_among_targets()
        self.poisoned_test.distribute_poison_indices_among_targets()
        
        return
    
    
    def prepare_localtion_for_each_target(self):
        
        # Pytorch __getitem__() gives an image in the following shape: [n_channels, n_rows, n_cols]
        n_rows, n_cols = self.item.detach().cpu().numpy().shape[1:]
        
        n_slices = np.sqrt(self.backdoor_configuration['num_targets'])
        n_slices = int(n_slices)+1 if int(n_slices)<n_slices else int(n_slices)
        self.n_slices = n_slices
        
        assert n_rows > n_slices, 'Not enough space in the image rows for the number of target locations.'
        assert n_cols > n_slices, 'Not enough space in the image cols for the number of target locations.'
        slice_size = min(n_rows // n_slices, n_cols // n_slices)
        trigger_size = min(slice_size, 5)
        
        self.triggers = []
        for i in range(self.backdoor_configuration['num_targets']):
            trigger = torch.zeros_like(self.item)
            trigger_start, trigger_end = (i // self.n_slices)*slice_size, (i % self.n_slices)*slice_size
            trigger[:, trigger_start:trigger_start+trigger_size, trigger_end:trigger_end+trigger_size] = -1
            trigger[min(1, len(trigger)-1), trigger_start:trigger_start+trigger_size, trigger_end:trigger_end+trigger_size] = 1
            self.triggers.append(trigger)
            
        return
    
    
    def poison(self, x, y, class_=0):
        min_value = 0 if torch.min(x)>0 else torch.min(x)
        max_value = 1 if torch.max(x)<1 else torch.max(x)
        return torch.clamp(x+self.triggers[class_]*(max_value-min_value), min_value, max_value), self.targets[class_]
    
    