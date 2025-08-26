import numpy as np
import torch
import copy

import matplotlib.pyplot as plt


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from .npca_custom_no_masking import NPCA_Custom

from utils_.pca import PCA_Loss, PCA_of_SKLEARN, PCA_of_NPCA
from utils_.torch_utils import get_outputs, get_data_samples_from_loader, evaluate_on_numpy_arrays, prepare_dataloader_from_numpy
from utils_.general_utils import normalize, exponential_normalize, np_sigmoid

from ..attacks.input_minimalist import Input_Minimalist
from ..attacks.fgsm_attack import FGSM_with_Dict
from ..attacks.input_minimalist_patch import Patch_Input_Minimalist
from ..attacks.adv_attack import Random_Patch_Adversarial_Attack



torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def input_minimalist_functions(
    minimalist_type: str, model: Torch_Model, minimalist_configuration: dict
) -> Input_Minimalist:
    
    all_input_minimalists = {
        'patch': Random_Patch_Adversarial_Attack,
        'fgsm': FGSM_with_Dict,
        'pixel_based': Input_Minimalist,
        'patch_based': Patch_Input_Minimalist
    }
    
    return all_input_minimalists[minimalist_type](model, minimalist_configuration)


class NPCA_Custom_with_Masking(NPCA_Custom):
    
    def __init__(
        self, 
        data: Torch_Dataset=None, 
        torch_model: Torch_Model=None, 
        defense_configuration: dict={},
        verbose: bool=True,
        **kwargs
    ):
        
        self.defense_name = 'SNPCA_Masking'
        
        default_defense_configuration = {
            'masking_configuration': {
                'alpha': 0.5,
                'mask_ratio': 0.8,
                'patch_size': 0.15,
                'iterations': 100,
            }
        }
        for key in default_defense_configuration.keys():
            if key not in defense_configuration.keys():
                defense_configuration[key] = default_defense_configuration[key]
        
        super().__init__(data=data, torch_model=torch_model, defense_configuration=defense_configuration, verbose=verbose)
        self.prepare_masked_data()
        
        return
    
    
    def prepare_masked_data(self):
        
        self.prepare_redefined_transformed_loader()
        
        _x, y = get_data_samples_from_loader(self.data_loader, return_numpy=True)
        perturbed_x = _x.copy()
        
        # fgsm_attack = input_minimalist_functions('fgsm', self.model, {'epsilon': 0.05})
        # fgsm_perturbations = fgsm_attack.attack(perturbed_x, self.target_class*np.ones_like(y), iterations=10, targeted=False)
        # perturbed_x = fgsm_attack.perturb(perturbed_x, -fgsm_perturbations)
        
        # mask_attack = input_minimalist_functions('pixel_based', self.model, {'mask_ratio': 0.2})
        # masks = mask_attack.attack(perturbed_x, self.target_class*np.ones_like(y), iterations=self.configuration['masking_configuration']['iterations'], targeted=True)
        # perturbed_x = mask_attack.perturb(perturbed_x, masks)
        
        mask_attack = input_minimalist_functions('patch_based', self.model, self.configuration['masking_configuration'])
        masks = mask_attack.attack(perturbed_x, self.target_class*np.ones_like(y), iterations=self.configuration['masking_configuration']['iterations'], targeted=False)
        perturbed_x = mask_attack.perturb(perturbed_x, -1 * masks)
        
        masked_data_loader = prepare_dataloader_from_numpy(perturbed_x.astype(np.float32), y, batch_size=self.data_loader.batch_size)
        
        self.configure_defense(data_loader=masked_data_loader)
        self.prepare_npca_things_original()
        
        return
    
    
    def prepare_npca_things_original(self):
        
        self.ac_original = get_outputs(self._model, self.data_loader_original)
        self._pca_original = self.pca_functions(self.ac_original.detach().cpu().numpy(), n_components=self.configuration['n_components'])
        
        return
    
    
    def prepare_redefined_transformed_loader(self):
        
        self.data.update_transforms(self.redefined_test_transform)
        _xq, y = get_data_samples_from_loader(self.data_loader, return_numpy=True)
        self.transformed_loader = prepare_dataloader_from_numpy(_xq, y, batch_size=self.data_loader.batch_size)
        self.data.reset_transforms()
        
        return