import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from copy import deepcopy

from _0_general_ML.data_utils.datasets import GTSRB, Fashion_MNIST
from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from ...backdoor_attacks.simple_backdoor import Simple_Backdoor

from .backdoor_defense import Backdoor_Detection_Defense

from utils_.torch_utils import get_data_samples_from_loader, get_outputs



class STRIP(Backdoor_Detection_Defense):
    """
    This code was modified from: https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master
    """
    
    def __init__(self, torch_model: Torch_Model=None, defense_configuration: dict={}, **kwargs):
        
        super().__init__(torch_model=torch_model, defense_configuration=defense_configuration)
        
        self.device = torch_model.device
        
        return
    

    def configure_defense(self, defense_configuration: dict={}):
        
        super().configure_defense(defense_configuration=defense_configuration)
        
        default_strip_configuration = {
            # 'number_of_clean_samples_for_computing_threshold': 100,
            'number_of_perturbations': 20,
            'threshold': 0.5,
        }
        for key in default_strip_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_strip_configuration[key]
        
        self.n_sample = self.configuration['number_of_perturbations']
        self.threshold = self.configuration['threshold']
        
        return
    
    
    def defend_mr(self, epochs: int=10, *args, **kwargs):
        self.torch_model.train(epochs=epochs)
        self.defend()
        return
    
    
    def defend(self, *args, **kwargs):
        
        self.prepare_defense()
        self.prepare_threshold()
        
        altered_model = Torch_Model(self.ood_data, self.torch_model.model_configuration, path='')
        altered_model.model.load_state_dict(self.torch_model.model.state_dict())
        altered_model.model = self.purify_model(altered_model.model)
        
        # self.final_model = deepcopy(altered_model)
        self.final_model = Torch_Model(altered_model.data, altered_model.model_configuration, path=altered_model.path)
        self.final_model.model = deepcopy(altered_model.model)
        
        return
    
    
    def prepare_defense(self, *args, **kwargs):
        
        self.available_data = self.torch_model.data.test
        
        if self.data_channels == 1:
            self.ood_data = Fashion_MNIST(preferred_size=self.preferred_size, data_means=self.torch_model.data.data_means, data_stds=self.torch_model.data.data_stds)
        else:
            self.ood_data = GTSRB(preferred_size=self.preferred_size, data_means=self.torch_model.data.data_means, data_stds=self.torch_model.data.data_stds)
        self.backgrounds = np.array([self.ood_data.test[k][0].numpy() for k in range(self.torch_model.model_configuration['batch_size'])])
        
        return
    
    
    def _superimpose(self, background, overlay):
        output = cv2.addWeighted(background, 1, overlay, 1, 0)
        if len(output.shape) == 2:
            output = np.expand_dims(output, 2)
        return output
    
    
    def _get_entropy(self, dataset, model=None):
        
        entropy_values = []
        for i in range(self.n_sample):
            # entropy_sum = [0] * self.n_sample
            # index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
            
            add_image = self._superimpose(
                self.backgrounds[np.random.choice(len(self.backgrounds), size=len(dataset), replace=False)], 
                dataset.detach().cpu().numpy()
            )
            add_image = torch.tensor(add_image).to(dataset.device)
            
            # x1_add = torch.tensor(x1_add)
            if model is None:
                py1_add = self.torch_model.model(add_image)
            else:
                py1_add = model(add_image)
            # py1_add = torch.sigmoid(py1_add)
            py1_add = py1_add.detach().cpu().numpy()
            entropy_values.append(py1_add)
        
        entropy_sum = -np.nansum(entropy_values * np.log2(entropy_values), axis=(0,2))
        
        return entropy_sum / self.n_sample

    
    def purify_model(self, model: torch.nn.Module=None, training_mode: bool=False, **kwargs):
        
        class Purified_Net(torch.nn.Module):
            
            def __init__(local_self, net, training_mode: bool=False):
                super().__init__()
                assert net is not None, 'model is None. Please pass a model to be purified for STRIP to work.'
                local_self.net = net
                local_self.training_mode = training_mode
                return
            
            
            def detect(local_self, dataset):
                # entropy = torch.tensor([self._get_entropy(dataset[i:i+1]) for i in range(len(dataset))])
                entropy = torch.tensor(self._get_entropy(dataset))
                return entropy, entropy<self.threshold
            
            
            def __call__(local_self, dataset):
                entropy, positives = local_self.detect(dataset)
                if local_self.training_mode:
                    return entropy.to(dataset.device)
                output = local_self.net(dataset.to(self.device))
                output_random = torch.normal(0, 10, size=output.shape).to(output.device)
                output[positives] = output_random[positives].clone()
                return output
            
            
        return Purified_Net(model, training_mode=training_mode)
    
    
    def prepare_threshold(self, num_samples: int=100):
        
        altered_model = Torch_Model(self.ood_data, self.torch_model.model_configuration, path='')
        altered_model.model.load_state_dict(self.torch_model.model.state_dict())
        altered_model.model = self.purify_model(altered_model.model, training_mode=True)
        
        clean_dataloader = torch.utils.data.DataLoader(self.available_data, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=True)
        clean_entropies = get_outputs(altered_model.model, clean_dataloader, return_numpy=True)
        clean_entropies = np.sort(clean_entropies)
        # clean_entropies = clean_entropies[np.random.randint(0, len(clean_entropies), size=num_samples)]
        
        self.threshold = clean_entropies[int(0.01*len(clean_entropies))]
        print(f'Prepared the threshold. The threshold is {self.threshold}. {np.mean(clean_entropies>self.threshold)}')
        
        return
    
    
    def __deprecated_get_entropy(self, dataset, model=None):
        
        entropy_sum = [0] * self.n_sample
        x1_add = [0] * self.n_sample
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        for index in range(self.n_sample):
            add_image = self._superimpose(self.backgrounds[np.random.randint(len(self.backgrounds))], dataset[index_overlay[index]].detach().cpu().numpy())
            x1_add[index] = torch.tensor(add_image)
            
        # x1_add = torch.tensor(x1_add)
        if model is None:
            py1_add = self.torch_model.model(torch.stack(x1_add).to(self.device))
        else:
            py1_add = model(torch.stack(x1_add).to(self.device))
        py1_add = torch.sigmoid(py1_add).detach().cpu().numpy()
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
        
        return entropy_sum / self.n_sample
    
    
    
    