import numpy as np
import torch


from _0_general_ML.model_utils.torch_model import Torch_Model
from _0_general_ML.data_utils.dataset_cards.mnist import MNIST
from _0_general_ML.data_utils.dataset_cards.fashion_mnist import Fashion_MNIST
from _0_general_ML.data_utils.dataset_cards.cifar10 import CIFAR10
from _0_general_ML.data_utils.dataset_cards.gtsrb import GTSRB

from .backdoor_defense import Backdoor_Detection_Defense

from utils_.torch_utils import get_outputs



class STRIP_Unofficial(Backdoor_Detection_Defense):
    
    def __init__(self, torch_model: Torch_Model, defense_configuration: dict={}):
        
        super().__init__(torch_model, defense_configuration=defense_configuration)
        
        return
    
    
    def configure_defense(self, defense_configuration: dict={}):
        
        super().configure_defense(defense_configuration)
        
        default_strip_configuration = {
            'perturbation_strength': 0.3,
            'number_of_perturbations': 20,
            # 'threshold': 0.5
        }
        for key in default_strip_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_strip_configuration[key]
        
        self.defense_parameters = None
        self.threshold = 1
        
        return
    
    
    def purify_model(self, model: torch.nn.Module, preparation_mode: bool=False):
        
        class Purified_Net(torch.nn.Module):
            
            def __init__(local_self, net: torch.nn.Module, preparation_mode: bool=False):
                
                super().__init__()
                
                local_self.net = net
                local_self.preparation_mode = preparation_mode
                if preparation_mode:
                    print('The model is in preparation model.')
                
                return
            
            def forward(local_self, x):
                
                min_value = min(0, torch.min(x))
                max_value = max(1, torch.max(x))
                perturbation_strength = self.configuration['perturbation_strength']
                epsilon = (max_value-min_value)*self.configuration['perturbation_strength']
                
                perturbations = self.get_perturbations(x)
                perturbations = perturbations*(max_value-min_value) + torch.mean(x)
                
                output_values_original = local_self.net(x)
                output_values_random = torch.normal(0, 10, size=output_values_original.shape).to(output_values_original.device)
                
                output_values = []
                for perturbation in perturbations:
                    
                    # _delta = perturbation.unsqueeze(0)-x
                    # _delta = torch.clamp(_delta, -epsilon, epsilon)
                    # output_classes = local_self.net(torch.clamp(perturbed_input, min_value, max_value)).detach().cpu()
                    
                    perturbed_input = (1-perturbation_strength)*x + perturbation_strength*perturbation.unsqueeze(0)
                    output_classes = local_self.net(perturbed_input).detach().cpu()
                    
                    output_values.append(output_classes)
                
                output_values = torch.stack(output_values, dim=-1)
                output_values = torch.sigmoid(output_values).numpy()
                entropy = np.array([-np.nansum(output_value_ * np.log2(output_value_)) for output_value_ in output_values]) / len(perturbations)
                
                output_values_original[entropy<self.threshold] = output_values_random[entropy<self.threshold].clone()
                # print(entropy.shape)
                # print(np.min(entropy), np.mean(entropy), np.max(entropy))
                # assert False
                
                if local_self.preparation_mode:
                    return torch.tensor(entropy).to(x.device)
                
                return output_values_original
        
        model = Purified_Net(model, preparation_mode=preparation_mode)
        
        return model
    
    
    def get_perturbations(self, x):
        
        perturbations = torch.stack(
            [self.ood_data.train[i][0] for i in np.random.choice(self.ood_data.train.__len__(), size=self.configuration['number_of_perturbations'])],
            dim=0
        ).to(x.device)
        
        return perturbations
    
    
    def prepare_defense(self, preferred_size: tuple[int, int]):
        
        # if self.defense_parameters is None:
        self.ood_data = GTSRB(preferred_size=preferred_size, data_means=[0.5], data_stds=[1])
        # self.ood_data = CIFAR10(preferred_size=preferred_size, data_means=[0.5], data_stds=[1])
        
        clean_loader = torch.utils.data.DataLoader(self.ood_data.test, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=True)
        self.model = self.purify_model(self.torch_model.model, preparation_mode=True)
        entropies = get_outputs(self.model, clean_loader, return_numpy=True)
        
        self.threshold = entropies[np.argsort(entropies)[int(0.05*len(entropies))]]
        print(f'Prepared the threshold. The threshold is {self.threshold}. {np.mean(entropies>self.threshold)}')
        
        return
    
    