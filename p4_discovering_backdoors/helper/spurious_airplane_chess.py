import numpy as np
import torch, torchvision
# import cv2
from termcolor import colored


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor


class Spurious_Airplane_Chess(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset,
        backdoor_configuration: dict={},
        model = None,
        **kwargs
    ):
        
        backdoor_configuration['target'] = 0
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration,
            attack_name='spurious_ba'
        )
        
        return
    
    
    def poison_data(self):
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
        
        if self.poison_ratio > 0:
            target_indices = np.where(self.train.targets==self.targets[0])[0]
            self.num_poison_samples = min(self.num_poison_samples, len(target_indices))
            self.poison_indices = list(np.random.choice(target_indices, size=self.num_poison_samples, replace=False))
            
            self.train.poison_indices = np.array(self.poison_indices)
            self.train.poisoner_fn = self.poison
            self.train.update_targets(self.train.poison_indices, [self.targets[0]]*len(self.train.poison_indices))
            
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison
        self.poisoned_test.update_targets(self.poisoned_test.poison_indices, [self.targets[0]]*len(self.poisoned_test.poison_indices))
        
        return
    
    