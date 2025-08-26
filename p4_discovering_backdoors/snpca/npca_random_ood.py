import numpy as np

import torch
import torchvision
import copy

import matplotlib.pyplot as plt

from termcolor import colored

from sklearn.cluster import SpectralClustering, KMeans, HDBSCAN
from sklearn.metrics import f1_score, accuracy_score


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

# from .npca_paper import NPCA_Paper
from .npca_random import NPCA_Random

from utils_.torch_utils import get_outputs, get_data_samples_from_loader, evaluate_on_numpy_arrays, prepare_dataloader_from_numpy
from utils_.general_utils import normalize, exponential_normalize

from ..attacks.input_minimalist import Input_Minimalist
from ..attacks.input_minimalist_patch import Patch_Input_Minimalist
from ..attacks.fgsm_attack import FGSM_with_Dict
from ..attacks.adv_attack import Random_Patch_Adversarial_Attack
from ..attacks.activation_based_pgd import Activation_Based_PGD
from ..attacks.adv_attack_copy import Random_Patch_Invisible_Visible_Adversarial_Attack



torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class NPCA_Random_OOD(NPCA_Random):
    
    def __init__(
        self, 
        data: Torch_Dataset,
        torch_model: Torch_Model, 
        defense_configuration: dict={},
        verbose: bool=True,
        **kwargs
    ):
        
        super().__init__(data=data, torch_model=torch_model, defense_configuration=defense_configuration, verbose=verbose)
        
        return
    
    
    def get_data_with_targets(self, my_data: Torch_Dataset, _type='test', redefine_transformation: bool=True):
        
        self.print_out(my_data.data_name, 'kaggle_imagenet' not in my_data.data_name)
        _data = my_data.train if _type=='train' else my_data.test
        
        if redefine_transformation:
            my_data.update_transforms(self.redefined_test_transform, subdata_category=_type)
            
        return _data
    
    
    def get_data_subset_personal(self, imagenet, class_id, bs=128, num_workers=0, **kwargs):
        
        from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
        
        self.print_out(colored('using random indices...', 'light_red'))
        _poison_in_ = np.where(torch.LongTensor(imagenet.targets) != self.target_class)[0]
        _poison_in_ = np.random.choice(_poison_in_, min(5000, len(_poison_in_)), replace=False)
        _poison_in = []
        for i in _poison_in_: 
            if i not in self.poison_indices: _poison_in.append(i)
            
        in_subset = Client_SubDataset(imagenet, _poison_in)
        loader = torch.utils.data.DataLoader(in_subset, batch_size=bs, shuffle=False, num_workers=num_workers)
        
        return in_subset, loader
    
    
    