import numpy as np
import torch


from _0_general_ML.model_utils.torch_model import Torch_Model

from .adversarial_attack import Adversarial_Attack



class PGD(Adversarial_Attack):
    
    def __init__(self, model: Torch_Model, loss='crossentropy', input_mask=None, output_mask=None, verbose: bool=True):
        super().__init__(model, loss=loss, input_mask=input_mask, output_mask=output_mask, verbose=verbose)
        return
    
    
    def attack(
        self, x_input, y_input,
        epsilon=0.1, norm='li', 
        epsilon_per_iteration=0.03,
        iterations=1000,
        targeted=False, verbose: bool=False,
        **kwargs
    ):
        
        self.last_run_loss_values = []
        epsilon *= np.max(x_input)-np.min(x_input)
        epsilon_per_iteration = epsilon/(iterations/4)

        x_perturbation = np.zeros_like(x_input).astype(np.float32)
        for iteration in range(iterations):
            pre_str = f'\rPGD Iteration {iteration+1}/{iterations} | '
            
            x_perturbation = self.fgsm_step(
                x_input, y_input, x_perturbation, 
                epsilon=epsilon_per_iteration,
                targeted=targeted,
                pre_str=pre_str
            )
            
            x_perturbation = np.clip(x_perturbation, -epsilon, epsilon)
            x_perturbation = np.clip(x_input+x_perturbation*self.input_mask, np.min(x_input), np.max(x_input)) - x_input
        self.print_out()
            
        return np.clip(x_input + x_perturbation, np.min(x_input), np.max(x_input))