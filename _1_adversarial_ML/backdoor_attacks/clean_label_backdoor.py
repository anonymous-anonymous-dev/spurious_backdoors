import os
import numpy as np
import torch
import copy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from ..adversarial_attacks.pgd import PGD

from .simple_backdoor import Simple_Backdoor

from utils_.torch_utils import get_data_samples_from_loader



class Clean_Label_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, 
        model = None,
        **kwargs
    ):
        
        self.model = model
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration,
            attack_name='clean_label'
        )
        
        return
    
    
    def configure_backdoor(self, backdoor_configuration, **kwargs):
        
        super().configure_backdoor(backdoor_configuration, **kwargs)
        
        default_backdoor_configuration = {
            'epsilon': 0.1,
            'iterations': 100,
            'use_model': False
        }
        for key in default_backdoor_configuration.keys():
            if key not in self.backdoor_configuration.keys():
                self.backdoor_configuration[key] = default_backdoor_configuration[key]
        
        non_target_indices = np.where(np.array(self.train.targets)!=self.targets[0])[0]
        non_target_loader = torch.utils.data.DataLoader(Client_SubDataset(self.train, non_target_indices), batch_size=self.backdoor_configuration['batch_size'], shuffle=True)
        self.non_target_perturbations, _ = get_data_samples_from_loader(non_target_loader)
        min_value = min(torch.min(self.non_target_perturbations), 0)
        max_value = max(torch.max(self.non_target_perturbations), 1)
        self.threshold = self.backdoor_configuration['epsilon'] * (max_value-min_value)
        
        return
    
    
    def poison_data(self):
        
        assert len(self.targets) == 1, 'Clean label backdoor only supports the poisoning for one single class at the moment.'
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
        
        if self.poison_ratio > 0:
            self.poison_indices = np.random.choice(
                np.where(self.train.targets==self.targets[0])[0],
                self.num_poison_samples,
                replace=True
            )
            
            self.train.poison_indices = self.poison_indices
            self.train.poisoner_fn = self.poison
            self.train.update_targets(self.train.poison_indices, [self.targets[0]]*len(self.train.poison_indices))
            
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison
        self.poisoned_test.update_targets(self.poisoned_test.poison_indices, [self.targets[0]]*len(self.poisoned_test.poison_indices))
        
        return
    
    
    def get_perturbation(self, x: torch.tensor):
        
        # assert isinstance(self.model, Torch_Model), 'The model must not be None and must be an instance of Torch_Model class. Please wrap it.'
        if self.backdoor_configuration['use_model']:
            if (self.model is not None) and (isinstance(self.model, Torch_Model)):
                attack = PGD(self.model)
                perturbation = attack.attack(np.expand_dims(x.detach().cpu().numpy(), axis=0), self.targets[0:1], targeted=False, iterations=5)[0]
            else:
                raise TypeError('The model must not be None and must be an instance of Torch_Model class. Please wrap it.')
            
        else:
            perturbation = self.non_target_perturbations[np.random.randint(len(self.non_target_perturbations))]
            
        perturbation = torch.clamp(perturbation-x, -self.threshold, self.threshold)
        
        return perturbation
    
    
    def __poison(self, x, y, index=None, **kwargs):
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x + self.perturbations[index] + self.triggers[0]*(max_value-min_value), min_value, max_value), self.targets[0]
    
    
    def poison(self, x, y, index=None, **kwargs):
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x + self.get_perturbation(x) + self.triggers[0]*(max_value-min_value), min_value, max_value), y #self.targets[0]
    
    