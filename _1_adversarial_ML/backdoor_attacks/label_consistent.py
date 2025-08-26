import os
import numpy as np
import torch, torchvision
import copy
from termcolor import colored
import gc


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.pgd import PGD

from _4_generative_ML.computer_vision.autoencoders.denoised_autoencoder import Denoising_Autoencoder

from ..adversarial_attacks.pgd import PGD

from .simple_backdoor import Simple_Backdoor
from .poisonable_class import Poisonable_Data

from utils_.torch_utils import get_data_samples_from_loader, get_outputs



class Label_Consistent_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, 
        model = None,
        **kwargs
    ):
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration,
            model=model,
            attack_name='lcba'
        )
        
        self.adversarial_recorder = 0
        
        return
    
    
    def configure_backdoor(self, backdoor_configuration, **kwargs):
        
        super().configure_backdoor(backdoor_configuration, **kwargs)
        
        default_backdoor_configuration = {
            'adversarial_setting': {
                'epsilon': 0.1,
                'iterations': 100,
                'adversarial_reset': 500    
            },
            'autoencoder_setting':{
                'noise_mag': 0, 
                'n_iter': 0,
                'device': 'cuda',
                'interpolation_mag': 0.4
            },
            'use_model': False
        }
        for key in default_backdoor_configuration.keys():
            if key not in self.backdoor_configuration.keys():
                self.backdoor_configuration[key] = default_backdoor_configuration[key]
        
        return
    
    
    def poison_data(self):
        
        assert len(self.targets) == 1, 'Clean label backdoor only supports the poisoning for one single class at the moment.'
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
        
        if self.poison_ratio > 0:
            target_indices = np.where(self.train.targets==self.targets[0])[0]
            self.num_poison_samples = min(self.num_poison_samples, len(target_indices))
            self.poison_indices = np.random.choice(target_indices, size=self.num_poison_samples, replace=False)
            
            if self.model is None:
                self.backdoor_mode = 'autoencoder'
                self.prepare_autoencoder_data()
            else:
                self.backdoor_mode = 'adversarial'
                self.prepare_adversarial_data()
            
            self.train.poison_indices = self.poison_indices
            self.train.poisoner_fn = self.poison_train
            self.train.update_targets(self.train.poison_indices, [self.targets[0]]*len(self.train.poison_indices))
            
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison
        self.poisoned_test.update_targets(self.poisoned_test.poison_indices, [self.targets[0]]*len(self.poisoned_test.poison_indices))
            
        return
    
    
    def update_model(self, model: Torch_Model):
        
        self.model = model
        self.backdoor_mode = 'adversarial'
        self.adversarial_recorder = 0
        self.prepare_adversarial_data()
        
        return
    
    
    def prepare_adversarial_data(self):
        
        if self.model is not None:
            
            # device = self.model.device
            # self.model.device = 'cpu'
            # self.model.model.to('cpu')
            # self.model.model.cpu()
            
            attack = PGD(self.model)
            x = np.array([self.train.data[i][0].numpy() for i in self.poison_indices])
            p_x = attack.attack(
                x, [self.targets[0]]*len(x), 
                epsilon=self.backdoor_configuration['adversarial_setting']['epsilon'], 
                iterations=self.backdoor_configuration['adversarial_setting']['iterations'], targeted=False,
                verbose=True
            )
            self.adversarial_perturbations = p_x - x
            
            # self.model.model.to(device)
            torch.cuda.empty_cache()
            gc.collect()
            
        else:
            print(colored('Using autoencoder as the model is None.', 'red'))
            self.prepare_autoencoder_data()
        
        # # analyze the available dataset
        # non_target_indices = np.where(np.array(self.parent_data.train.targets)!=self.targets[0])[0]
        # non_target_indices = np.random.choice(non_target_indices, size=min(200, len(non_target_indices)))
        # non_target_loader = torch.utils.data.DataLoader(Client_SubDataset(self.parent_data.train, non_target_indices), batch_size=self.backdoor_configuration['batch_size'], shuffle=True)
        # self.non_target_perturbations, _ = get_data_samples_from_loader(non_target_loader)
        
        # # prepare perturbations for adversarial setting
        # min_value = min(torch.min(self.non_target_perturbations), 0)
        # max_value = max(torch.max(self.non_target_perturbations), 1)
        # self.threshold = self.backdoor_configuration['adversarial_setting']['epsilon'] * (max_value-min_value)
            
        return
    
    
    def prepare_autoencoder_data(self):
        
        # prepare things for autoencoder setting
        self.dae = Denoising_Autoencoder(configuration=self.backdoor_configuration['autoencoder_setting'])
        
        # analyze the available dataset
        non_target_indices = np.where(np.array(self.parent_data.train.targets)!=self.targets[0])[0]
        non_target_indices = np.random.choice(non_target_indices, size=min(200, len(non_target_indices)))
        non_target_loader = torch.utils.data.DataLoader(Client_SubDataset(self.parent_data.train, non_target_indices), batch_size=self.backdoor_configuration['batch_size'], shuffle=True)
        non_target_perturbations, _ = get_data_samples_from_loader(non_target_loader)
        # computing normalization values
        min_ = min(0, torch.min(non_target_perturbations))
        max_ = max(1, torch.max(non_target_perturbations))
        non_target_perturbations = (non_target_perturbations-min_)/(max_-min_)
        # compute encodings of the target class samples
        target_inputs = torch.stack([self.train.data[i][0] for i in self.poison_indices], dim=0)
        print(target_inputs.shape, '#############################')
        if self.data_name=='mnist':
            non_target_perturbations = torch.cat([non_target_perturbations]*3, dim=1)
            target_inputs = torch.cat([target_inputs]*3, dim=1)
        
        # compute encodings of non target class samples
        self.non_target_encodings = self.dae.encoder_foward(non_target_perturbations).detach().cpu()
        torch.cuda.empty_cache()
        gc.collect()
        
        # target_inputs = (target_inputs_-min_)/(max_-min_)
        encoded_target_inputs = self.dae.encoder_foward(target_inputs).detach().cpu()
        torch.cuda.empty_cache()
        gc.collect()
        
        # interpolate and compute new inputs and denormalize
        random_indices = np.random.choice(len(self.non_target_encodings), size=len(self.poison_indices), replace=True)
        interpolated_x = encoded_target_inputs + self.backdoor_configuration['autoencoder_setting']['interpolation_mag']*(self.non_target_encodings[random_indices] - encoded_target_inputs)
        new_inputs = self.dae.decoder_forward(interpolated_x).detach().cpu()
        torch.cuda.empty_cache()
        gc.collect()
        
        # resizeing new inputs to the preferred_size shape
        if new_inputs.shape[1:] != target_inputs.shape[1:]:
            new_inputs = torch.stack([torchvision.transforms.functional.resize(inp_, list(self.preferred_size)) for inp_ in new_inputs], dim=0)
        
        new_inputs = torch.clamp(new_inputs, 0, 1)
        # min_n = min(0, torch.min(new_inputs))
        # max_n = max(1, torch.max(new_inputs))
        new_inputs = new_inputs * (max_-min_) + min_
        
        # prepare perturbations
        self.perturbations_autoencoder = torch.zeros([self.train.__len__()]+list(new_inputs.shape[1:]))#.astype(np.float32)
        print(interpolated_x.shape, new_inputs.shape)
        self.perturbations_autoencoder[self.poison_indices] = new_inputs - target_inputs
        if self.data_name=='mnist':
            self.perturbations_autoencoder = torch.mean(self.perturbations_autoencoder, dim=1, keepdims=True)
        
        return
    
    
    def get_perturbation(self, x: torch.tensor, index: int):
        
        # assert isinstance(self.model, Torch_Model), 'The model must not be None and must be an instance of Torch_Model class. Please wrap it.'
        
        if self.backdoor_mode == 'adversarial':
            # if self.backdoor_configuration['use_model']:
            #     if (self.model is not None) and (isinstance(self.model, Torch_Model)):
            #         attack = PGD(self.model)
            #         perturbation = attack.attack(np.expand_dims(x.detach().cpu().numpy(), axis=0), self.targets[0:1], targeted=False, iterations=5)[0]
            #     else:
            #         raise TypeError('The model must not be None and must be an instance of Torch_Model class. Please wrap it.')
                
            # else:
            #     perturbation = self.non_target_perturbations[np.random.randint(len(self.non_target_perturbations))]
            # perturbation = torch.clamp(perturbation-x, -self.threshold, self.threshold)
            if self.adversarial_recorder == 0:
                self.prepare_adversarial_data()
            perturbation = self.adversarial_perturbations[list(self.poison_indices).index(index)]
            self.adversarial_recorder = (self.adversarial_recorder+1)%self.backdoor_configuration['adversarial_setting']['adversarial_reset']
            
        else:
            # random_index = np.random.randint(len(self.non_target_perturbations))
            # encoded_x = self.dae.encoder_foward(x.unsqueeze(0)).detach().cpu()
            # perturbation = self.non_target_encodings[random_index:random_index+1]
            # interpolated_x = encoded_x + self.backdoor_configuration['autoencoder_setting']['interpolation_mag']*(perturbation - encoded_x)
            # new_x = self.dae.decoder_forward(interpolated_x).detach().cpu()
            # perturbation = new_x[0] - x
            perturbation = self.perturbations_autoencoder[index]
        
        return perturbation
    
    
    def __poison(self, x, y, index=None, **kwargs):
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x + self.perturbations[index] + self.triggers[0]*(max_value-min_value), min_value, max_value), self.targets[0]
    
    
    def poison_train(self, x, y, index=None, **kwargs):
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x+self.get_perturbation(x, index) + self.triggers[0]*(max_value-min_value), min_value, max_value), self.targets[0]
    
    