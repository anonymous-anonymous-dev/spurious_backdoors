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


class Adversarial_Attack:
    
    def __init__(
        self, model: Torch_Model, loss='crossentropy', loss_functions: dict={},
        input_mask=None, output_mask=None,
        verbose: bool=True
    ):
        
        self.model = model.model
        self.device = model.device
        self.loss = loss
        self.batch_size = model.model_configuration['batch_size']
        
        self.loss_functions = local_loss_functions
        
        self.input_mask = 1 #np.ones_like(model.data.train.__getitem__(0)[0])
        if input_mask is not None:
            self.input_mask = input_mask
        
        self.output_mask = np.ones_like(model.data.train.__getitem__(0)[1])
        if output_mask is not None:
            self.output_mask = output_mask
        
        self.verbose = verbose
        self.last_run_loss_values = []
        
        return
    
    
    def print_out(self, *args, verbose: bool=False, **kwargs):
        if not verbose:
            if self.verbose:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)
        return
    
    
    def adv_loss_outputs(self, y_true, y_pred):
        # y_pred = torch.exp(y_pred)/torch.sum(torch.exp(y_pred), dim=1, keepdims=True)
        # return -1 * torch.mean( y_true * torch.log(y_pred) )
        # return torch.mean(torch.square(y_true - y_pred))
        return self.loss_functions[self.loss](y_true, y_pred)
    
    
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
            self.print_out(f'\r{pre_str} | Running batch: {i+1}/{n_batches}', end='')
            _x_delta = self.step(
                x_input[i*self.batch_size:(i+1)*self.batch_size], y_input[i*self.batch_size:(i+1)*self.batch_size],
                x_perturbation[i*self.batch_size:(i+1)*self.batch_size], epsilon=epsilon, targeted=targeted
            )
            x_delta.append(_x_delta)
            
        grads_sign = torch.cat(x_delta, dim=0).sign().numpy()
        
        return (x_perturbation - epsilon*grads_sign*self.input_mask)
    
    
    def step(
        self,
        x_input, y_input, x_perturbation,
        epsilon=0.01, targeted=False
    ):
        
        x_v = torch.tensor(x_input).to(self.device)
        y_in = torch.tensor(y_input).to(self.device) if isinstance(y_input, np.ndarray) else y_input.to(self.device)
        x_delta = torch.autograd.Variable(torch.tensor(x_perturbation)).to(self.device)
        x_delta.requires_grad = True
        
        prediction = self.model(x_v+x_delta)
        
        if targeted:
            loss = self.adv_loss_outputs(prediction, y_in)
        else:
            loss = -1 * self.adv_loss_outputs(prediction, y_in)
        
        # loss.requires_grad = True
        self.model.zero_grad()
        torch.mean(loss).backward()
        
        self.last_run_loss_values += [torch.mean(loss).item()]
        
        # grads_sign = x_delta.grad.data.sign()
        return x_delta.grad.data.cpu()
        # return (x_perturbation - epsilon*grads_sign.cpu().numpy()*self.input_mask)
    
    
    def batch_wise_fgsm_step(
        self,
        x_input, y_input, x_perturbation,
        epsilon=0.01, targeted=False
    ):
        
        no_of_batches = int(len(x_input) / self.batch_size) + 1
        
        loss_over_inputs = 0
        grad_sign = np.zeros_like(x_perturbation)
        for batch_number in range(no_of_batches):
            start_index = batch_number * self.batch_size
            end_index = min( (batch_number+1)*self.batch_size, len(x_input) )
            
            x_v = torch.tensor(x_input[start_index:end_index]).to(self.device)
            y_in = torch.tensor(y_input[start_index:end_index]).to(self.device)
            x_delta = torch.autograd.Variable(
                torch.tensor(x_perturbation[start_index:end_index])
            ).to(self.device)
            x_delta.requires_grad = True
            
            prediction = self.model(x_v+x_delta)
            
            if targeted:
                loss = self.adv_loss_outputs(prediction, y_in)
            else:
                loss = -1 * self.adv_loss_outputs(prediction, y_in)
            
            self.model.zero_grad()
            torch.mean(loss).backward()
            
            grad_sign[start_index:end_index] = x_delta.grad.data.sign()
            loss_over_inputs += torch.mean(loss).item()
        
        self.last_run_loss_values += [loss_over_inputs/len(x_input)]
        
        return (x_perturbation - epsilon*grad_sign.detach().cpu().numpy()*self.input_mask)
    
    
    def _deprecated_fgsm_step(
        self,
        data, x_perturbation,
        epsilon=0.01, target=None
    ):
        
        batch_size = 64
        data_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=batch_size)
        delta_perturbation = np.zeros_like(x_perturbation)
        
        loss_over_data = 0
        for idx, (data_in, data_out) in enumerate(data_loader):
            
            x_v, y_in = data_in.to(self.device), data_out.to(self.device)
            if target:
                y_in = torch.tensor(target[idx*batch_size: (idx+1)*batch_size]).to(self.device)
        
            x_delta = x_perturbation[idx*batch_size: (idx+1)*batch_size]
            x_delta = torch.autograd.Variable(torch.tensor(x_delta.astype(np.float32))).to(self.device)
            x_delta.requires_grad = True
            
            prediction = self.model(x_v + x_delta)
            
            if target:
                loss = self.adv_loss_outputs(prediction, y_in)
            else:
                loss = -1 * self.adv_loss_outputs(prediction, y_in)
            
            self.model.zero_grad()
            loss.backward()
            
            loss_over_data += loss.data
            delta_perturbation[idx*batch_size: (idx+1)*batch_size] = x_delta.grad.data.sign()
            
        self.last_run_loss_values += [loss_over_data/len(data.__len__())]
        
        return (x_perturbation - epsilon *delta_perturbation.detach().cpu().numpy()*self.input_mask)
    
    