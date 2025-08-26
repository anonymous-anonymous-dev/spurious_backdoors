import numpy as np
from copy import deepcopy
import torch, torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from ...backdoor_attacks.backdoor_data import Simple_Backdoor



class Backdoor_Detection_Defense:
    
    def __init__(self, torch_model: Torch_Model=None, defense_configuration: dict={}, verbose: bool=False, **kwargs):
        
        # super().__init__()
        
        self.verbose = verbose
        
        self.torch_model = torch_model
        self.data_channels = 1 if torch_model.data.data_name in ['mnist', 'fashion_mnist'] else 3
        self.preferred_size = torch_model.data.preferred_size
        self.configure_defense(defense_configuration=defense_configuration)
        
        return
    
    
    def configure_defense(self, *args, defense_configuration: dict={}, **kwargs):
        self.configuration = deepcopy(defense_configuration)
        return
    
    
    def defend(self, *args, **kwargs): self.final_model = deepcopy(self.torch_model); return
    def defend_mr(self, data_in: Simple_Backdoor, epochs: int=10, *args, **kwargs): 
        self.torch_model.train(epochs=epochs)
        self.defend(data_in)
        return
    def inference(self, x): return self.torch_model.model(x)
    
    
    def not_implemented_error(self, *args, **kwargs):
        raise NotImplementedError('This is the parent class. Please call the child class for things to work.')
    def defended_inference(self, x): return self.not_implemented_error()
    
    
    def print_out(self, *args, verbose: bool=False, **kwargs):
        if not verbose:
            if self.verbose:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)
        return
    
    
    def evaluate(self, data_in: Simple_Backdoor, *args, **kwargs):

        clean_dataloader = torch.utils.data.DataLoader(data_in.test, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False)
        poisoned_dataloader = torch.utils.data.DataLoader(data_in.poisoned_test, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False)
        
        loss_clean, acc_clean, _ = self.final_model.test_shot(clean_dataloader); print()
        loss_poisoned, acc_poisoned, _ = self.final_model.test_shot(poisoned_dataloader)
        
        return (loss_clean, acc_clean), (loss_poisoned, acc_poisoned)
    
    
    