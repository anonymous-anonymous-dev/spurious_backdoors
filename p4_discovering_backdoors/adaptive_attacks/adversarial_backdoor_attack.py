import os
import numpy as np
import torch
from copy import deepcopy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.ifgsm import i_FGSM
from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor
from utils_.torch_utils import get_data_samples_from_loader



class Adversarial_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, **kwargs
    ):
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration,
            attack_name='vtba'
        )
        
        return
    
    
    def adaptively_poison_data(self, model_torch: Torch_Model, epsilon: float=0.1, iterations: int=20):
        self.original_poisoned_test = deepcopy(self.poisoned_test)
        self.poisoned_test = self.get_adversarial_dataset(self.original_poisoned_test, model_torch, epsilon=epsilon, iterations=iterations)
        return
    
    
    def get_adversarial_dataset(
        self, 
        dataset: torch.utils.data.TensorDataset, 
        model_torch: Torch_Model, 
        epsilon: float=0.05, iterations: int=20
    ):
        
        batch_size=model_torch.model_configuration['batch_size']
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        x, y = get_data_samples_from_loader(dl, return_numpy=True)
        adv_y = np.ones_like(y).astype('int')
        
        mask = np.ones_like(x[:1]); mask[:, :, :10, :10]=0.
        attack_adv = i_FGSM(model_torch, input_mask=mask)
        adv_x = attack_adv.attack(x, adv_y, epsilon=epsilon, targeted=True, iterations=iterations)
        adv_y[:len(adv_y)//2] = y[:len(adv_y)//2].copy()
        
        return torch.utils.data.TensorDataset(torch.tensor(adv_x), torch.tensor(adv_y))
    
    