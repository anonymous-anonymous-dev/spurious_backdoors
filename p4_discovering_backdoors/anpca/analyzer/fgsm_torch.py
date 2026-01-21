import numpy as np
import torch


from _0_general_ML.model_utils.torch_model import Torch_Model



loss_function_reduction = 'none'
local_loss_functions = {
    'crossentropy': torch.nn.CrossEntropyLoss(reduction=loss_function_reduction),
    'mse': torch.nn.MSELoss(reduction=loss_function_reduction),
    'nll': torch.nn.NLLLoss(reduction=loss_function_reduction),
    'l1': torch.nn.L1Loss(reduction=loss_function_reduction),
    'kl_div': torch.nn.KLDivLoss(reduction=loss_function_reduction),
    'binary_crossentropy': torch.nn.BCELoss(reduction=loss_function_reduction),
}


class FGSM_Torch:
    
    def __init__(self, model: Torch_Model, loss='crossentropy', input_mask=None, output_mask=None, **kwargs):
        
        self.model = model.model
        self.device = model.device
        self.loss = loss
        self.batch_size = model.model_configuration['batch_size']
        
        self.loss_functions = local_loss_functions
        
        self.input_mask = np.ones_like(model.data.train.__getitem__(0)[0])
        if input_mask is not None:
            self.input_mask = input_mask
        
        self.output_mask = np.ones_like(model.data.train.__getitem__(0)[1])
        if output_mask is not None:
            self.output_mask = output_mask
        
        self.last_run_loss_values = []
        
        return
    
    
    def attack(self, x_input, y_input, epsilon=0.1, targeted=False, **kwargs):
        
        self.last_run_loss_values = []
        x_perturbation = torch.zeros_like(x_input)
        x_perturbation = self.fgsm_step(x_input, y_input, x_perturbation, epsilon=epsilon, targeted=targeted)
        
        return torch.clamp(x_input + x_perturbation, torch.min(x_input), torch.max(x_input))
    
    
    def fgsm_step(
        self,
        x_input, y_input, x_perturbation,
        epsilon=0.1, targeted=False,
        pre_str: str=''
    ):
        
        n_batches = int(len(x_input) // self.batch_size)
        n_batches += 1 if (n_batches*self.batch_size)<len(x_input) else 0
        
        x_delta = []
        for i in range(n_batches):
            print(f'\r{pre_str} | Running batch: {i+1}/{n_batches}', end='')
            _x_delta = self.step(
                x_input[i*self.batch_size:(i+1)*self.batch_size], y_input[i*self.batch_size:(i+1)*self.batch_size],
                x_perturbation[i*self.batch_size:(i+1)*self.batch_size], epsilon=epsilon, targeted=targeted
            )
            x_delta.append(_x_delta)
            
        grads_sign = torch.cat(x_delta, dim=0).sign()
        
        return (x_perturbation - epsilon*grads_sign*self.input_mask)
    
    
    def step(
        self,
        x_input, y_in, x_perturbation,
        epsilon=0.01, targeted=False
    ):
        
        x_input = x_input.to(self.device); y_in = y_in.to(self.device)
        x_delta = torch.autograd.Variable(x_perturbation).to(self.device)
        x_delta.requires_grad = True
        
        prediction = self.model(x_input+x_delta)
        
        if targeted:
            loss = self.adv_loss_outputs(prediction, y_in)
        else:
            loss = -1 * self.adv_loss_outputs(prediction, y_in)
        
        self.model.zero_grad()
        torch.mean(loss).backward()
        
        self.last_run_loss_values += [torch.mean(loss).item()]
        
        # grads_sign = x_delta.grad.data.sign()
        return x_delta.grad.data.cpu()
        # return (x_perturbation - epsilon*grads_sign.cpu().numpy()*self.input_mask)
    
    
    def adv_loss_outputs(self, y_true, y_pred):
        return self.loss_functions[self.loss](y_true, y_pred)
    
    
    