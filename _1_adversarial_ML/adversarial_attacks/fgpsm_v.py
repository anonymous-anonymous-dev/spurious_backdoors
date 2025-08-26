import numpy as np
import torch


from _0_general_ML.model_utils.torch_model import Torch_Model

from .adversarial_attack import Adversarial_Attack



class FGSM_V(Adversarial_Attack):
    
    def __init__(self, model: Torch_Model, loss='crossentropy', input_mask=None, output_mask=None, **kwargs):
        super().__init__(model, loss=loss, input_mask=input_mask, output_mask=output_mask)
        return
    
    
    def attack(self, x_input, y_input, epsilon=0.1, iterations: int=10, targeted=False, **kwargs):
        
        max_in = max(np.max(x_input), 1)
        min_in = min(np.min(x_input), 0)
        epsilon *= (max_in-min_in)
        epsilon_per_iteration = epsilon / iterations
        
        self.last_run_loss_values = []
        x_perturbation = np.zeros_like(x_input).astype(np.float32)
        for i in range(iterations):
            print(f'\rIteration {i+1}/{iterations}', end='')
            x_perturbation = self.fgsm_step(x_input, y_input, x_perturbation, epsilon=epsilon_per_iteration, targeted=targeted)
            x_perturbation = np.mean(x_perturbation, axis=0, keepdims=True)
            x_perturbation = np.clip(x_input+x_perturbation, min_in, max_in) - x_input
        
        return np.clip(x_input + x_perturbation, min_in, max_in)
    
    
    def fgsm_step(
        self,
        x_input, y_input, x_perturbation,
        epsilon=0.01, targeted=False
    ):
        
        def linearly_normalized_torch(arr_in: torch.Tensor):
            return (arr_in-torch.min(arr_in))/(torch.max(arr_in)-torch.min(arr_in)) if torch.max(arr_in)>torch.min(arr_in) else arr_in/torch.max(arr_in)
        
        
        x_v = torch.tensor(x_input).to(self.device)
        y_in = torch.tensor(y_input).to(self.device)
        x_delta = torch.autograd.Variable(torch.tensor(x_perturbation)).to(self.device)
        x_delta.requires_grad = True
        
        prediction = self.model(x_v+x_delta)
        
        if targeted:
            loss_1 = self.adv_loss_outputs(prediction, y_in)
        else:
            loss_1 = -1 * self.adv_loss_outputs(prediction, y_in)
        loss_2 = torch.mean(torch.abs(prediction-torch.mean(prediction, dim=0, keepdim=True)), dim=1)
        loss = linearly_normalized_torch(loss_1) + linearly_normalized_torch(loss_2)
        
        self.model.zero_grad()
        torch.mean(loss).backward()
        
        grads_sign = x_delta.grad.data.sign().cpu().numpy()
        
        self.last_run_loss_values += [torch.mean(loss).item()]
        
        return (x_perturbation - epsilon*grads_sign*self.input_mask)
    
    