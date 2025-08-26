import numpy as np
import torch, torchvision


from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.fgsm import FGSM
from _1_adversarial_ML.adversarial_attacks.ifgsm import i_FGSM
from _1_adversarial_ML.adversarial_attacks.pgd import PGD



class FGSM_with_Dict(i_FGSM):
    
    def __init__(
        self, 
        model: Torch_Model, 
        inversion_configuration: dict={}, 
        input_mask=None, output_mask=None,
        **kwargs
    ):
        
        self.configuration = {
            'loss': 'crossentropy',
            'epsilon': 0.1
        }
        for key in inversion_configuration:
            self.configuration[key] = inversion_configuration[key]
        
        super().__init__(model, self.configuration['loss'], input_mask, output_mask)
        
        self.rotation = torchvision.transforms.RandomRotation(20)
        
        return
    
    
    def fgsm_step_shot(
        self,
        x_input, y_input, x_perturbation,
        epsilon: float=0.01, targeted=False
    ):
        
        x_v = torch.tensor(x_input).to(self.device)
        y_in = torch.tensor(y_input).to(self.device)
        x_delta = torch.autograd.Variable(torch.tensor(x_perturbation)).to(self.device)
        x_delta.requires_grad = True
        
        # input
        inputs_to_model = x_v+x_delta
        # # # rotate inputs randomly
        # # rotate_randoms = np.random.uniform(-20, 20, size=len(inputs_to_model))
        # # inputs_rotated = torch.cat([torchvision.transforms.functional.rotate(inputs_to_model[i:i+1], rotate_randoms[i]) for i in range(len(inputs_to_model))], 0)
        inputs_to_model = self.rotation(inputs_to_model)
        
        prediction = self.model(inputs_to_model)
        
        if targeted:
            loss = self.adv_loss_outputs(prediction, y_in)
        else:
            loss = -1 * self.adv_loss_outputs(prediction, y_in)
        
        self.model.zero_grad()
        torch.mean(loss).backward()
        
        self.last_run_loss_values += [torch.mean(loss).item()]
        
        # x_perturbation -= epsilon * torch.cat(x_delta.grad.data.detach().cpu(), 0).sign().numpy()
        
        return x_delta.grad.data.detach().cpu()
    
    
    def fgsm_step(
        self, x_input, y_input, x_perturbation,
        epsilon: float=0.01, targeted=False
    ):
        
        no_of_batches = int(len(x_input) / self.batch_size) + 1
        
        x_delta_s, x_mask_s, loss_s = [], [], []
        for batch_number in range(no_of_batches):
            start_index = batch_number * self.batch_size
            end_index = min( (batch_number+1)*self.batch_size, len(x_input) )
            
            x_delta_grad = self.fgsm_step_shot(
                x_input[start_index:end_index], y_input[start_index:end_index], 
                x_perturbation[start_index:end_index], epsilon=epsilon, targeted=targeted
            )
            
            x_delta_s.append(x_delta_grad)
        
        x_perturbation -= epsilon * torch.cat(x_delta_s, 0).sign().numpy() * self.input_mask
        
        # self.last_run_loss_values.append(torch.mean(torch.stack(loss_s, 0)).item())
        
        return x_perturbation
    
    
    def attack(self, x_input, y_input, iterations=100, targeted: bool=True, **kwargs):
        x_perturbed = super().attack(x_input, y_input, epsilon=self.configuration['epsilon'], targeted=targeted, iterations=iterations, **kwargs)
        return x_perturbed-x_input
    
    
    def __attack(self, x_input, y_input, iterations=100, targeted: bool=True, **kwargs):
        
        self.last_run_loss_values = []
        epsilon *= np.max(x_input)-np.min(x_input)
        epsilon_per_iteration = epsilon/iterations
        x_perturbation = np.zeros_like(x_input).astype(np.float32)

        for iteration in range(iterations):
            print(f'\rIterations: {iteration}/{iterations}', end='')
            
            x_perturbation = self.fgsm_step(
                x_input, y_input, x_perturbation, 
                epsilon=epsilon_per_iteration,
                targeted=targeted
            )
            x_perturbation = np.clip(x_input+x_perturbation, np.min(x_input), np.max(x_input)) - x_input
            
        x_perturbed = np.clip(x_input + x_perturbation, np.min(x_input), np.max(x_input))
        
        return x_perturbed-x_input
    
    
    def perturb(self, x, masks):
        return np.clip(x + masks, np.min(x), np.max(x))
    
    