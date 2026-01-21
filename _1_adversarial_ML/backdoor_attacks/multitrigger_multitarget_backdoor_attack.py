import numpy as np
from sklearn.utils import shuffle
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from .multitarget_poisonable_data_class import Multi_Target_Poisonable_Data
from .simple_backdoor import Simple_Backdoor



class MultiTrigger_MultiTarget_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, **kwargs
    ):
        
        super().__init__(data, backdoor_configuration=backdoor_configuration, attack_name='mtmtba')
        
        self.prepare_localtion_for_each_target()
        self.poison_data()
        
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
        
        return
    
    
    def poison_data(self):
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
            
        if self.poison_ratio > 0:
            self.poison_indices = np.random.choice(self.train.__len__(), self.num_poison_samples, replace=True)
            assert len(self.poison_indices) >= self.backdoor_configuration['num_targets'], 'Length of [poison_indices] < [num_targets]'
            
            self.train.poison_indices = self.poison_indices
            self.train.poisoner_fn = self.poison
            self.train.distribute_poison_indices_among_targets()
        
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison
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
        
        self.trigger_masks = []
        for i in range(len(self.targets)):
            r = (i // self.n_slices) * slice_size
            c = (i % self.n_slices) * slice_size
            
            mask = torch.zeros_like(self.item)
            mask[:, r:r+trigger_size, :] += 0.5
            mask[:, :, c:c+trigger_size] += 0.5
            mask[mask < 0.9] = 0
            mask[mask >= 0.9] = 1.
            
            self.trigger_masks.append(mask)
        
        trigger_mark = -1 * torch.ones_like(self.item)
        trigger_mark[min(1, len(trigger_mark)-1)] = 1.
        self.triggers = [trigger_mark*self.trigger_masks[k] for k in range(len(self.trigger_masks))]
        # triggers = [0.8*torch.normal(0, 1, size=self.item.shape) for i in range(len(self.targets))]
        # self.triggers = [triggers[k]*self.trigger_masks[k] for k in range(len(triggers))]
        
        assert len(self.triggers) == self.backdoor_configuration['num_targets'], f'Len of triggers is {len(self.triggers)} while num_targets is {self.backdoor_configuration['num_targets']}'
        assert len(self.targets) == self.backdoor_configuration['num_targets'], f'Len of targets is {len(self.targets)} while num_targets is {self.backdoor_configuration['num_targets']}'
        
        return
    
    
    def poison(self, x, y, class_=0, **kwargs):
        min_value = 0 if torch.min(x)>0 else torch.min(x)
        max_value = 1 if torch.max(x)<1 else torch.max(x)
        return torch.clamp(x+self.triggers[class_]*(max_value-min_value), min_value, max_value), self.targets[class_]
    
    