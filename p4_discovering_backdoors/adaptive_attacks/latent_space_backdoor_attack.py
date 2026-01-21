import os
import numpy as np
import torch
from copy import deepcopy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model
from _0_general_ML.model_utils.generalized_model_activations_wrapper import Dependable_Feature_Activations

from _1_adversarial_ML.adversarial_attacks.ifgsm import i_FGSM
from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor
from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy, get_outputs



class Latent_Space_Backdoor(Simple_Backdoor):
    
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
    
    
    def adaptively_poison_data(self, model_torch: Torch_Model, clean_x: np.ndarray, epsilon: float=0.1, iterations: int=20):
        self.original_poisoned_test = deepcopy(self.poisoned_test)
        self.poisoned_test = self.get_adversarial_dataset(self.original_poisoned_test, model_torch, clean_x, epsilon=epsilon, iterations=iterations)
        return
    
    
    def get_adversarial_dataset(
        self, 
        dataset: torch.utils.data.TensorDataset, model_torch: Torch_Model, clean_x: np.ndarray, 
        epsilon: float=0.2, iterations: int=50
    ):
        
        batch_size=model_torch.model_configuration['batch_size']
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        x, y = get_data_samples_from_loader(dl, return_numpy=True)
        
        get_latent_layer_model = Dependable_Feature_Activations(model_torch, layer_numbers=np.arange(-15, 0), target_class=None, detach_output=False)
        
        new_model = Torch_Model(self, {})
        new_model.model = get_latent_layer_model
        
        outputs_groundtruth = get_outputs(get_latent_layer_model, prepare_dataloader_from_numpy(clean_x, np.zeros((len(clean_x))).astype('int'), batch_size=batch_size), return_numpy=True)
        outputs_groundtruth = np.mean(outputs_groundtruth, axis=0, keepdims=True)
        outputs_groundtruth = np.zeros([len(x)]+list(outputs_groundtruth[0].shape)) + outputs_groundtruth
        outputs_groundtruth = outputs_groundtruth.astype(np.float32)
        
        mask = np.ones_like(x[:1]); mask[:, :, :10, :10]=0.
        attack_adv = i_FGSM(new_model, loss='mse')
        adv_x = attack_adv.attack(x, outputs_groundtruth, epsilon=epsilon, targeted=True, iterations=iterations)
        
        return torch.utils.data.TensorDataset(torch.tensor(adv_x), torch.tensor(y))
    
    