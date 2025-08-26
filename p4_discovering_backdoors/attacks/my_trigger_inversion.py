import numpy as np
import torch
import gc


from _0_general_ML.model_utils.torch_model import Torch_Model

from _0_general_ML.model_utils.optimizer_utils.torch_optimizer import Torch_Optimizer

from _1_adversarial_ML.adversarial_attacks.trigger_inversion import Trigger_Inversion



class My_Trigger_Inversion(Trigger_Inversion):
    
    def __init__(
        self, 
        model: Torch_Model,
        trigger_inversion_configuration: dict={},
        input_mask=None, output_mask=None, 
        **kwargs
    ):
        
        self.trigger_inversion_configuration = {
            'alpha': 5e-3,
            'loss': 'crossentropy',
            'gray_scale_mask': False,
            'mask_ratio': 0.,
            'mask_perception': 1.,
            'mask_max': 20,
            'mask_min': -20
        }
        for key in trigger_inversion_configuration.keys():
            self.trigger_inversion_configuration[key] = trigger_inversion_configuration[key]
            
        super().__init__(model, loss=self.trigger_inversion_configuration['loss'], input_mask=input_mask, output_mask=output_mask)
        
        self.optimizer = Torch_Optimizer(name='sgd', lr=1e-3, momentum=0.5)
        
        self.alpha = np.clip(self.trigger_inversion_configuration['alpha'], 0, 1)
        self.update_rate = 1.
        
        # mp = np.clip(self.trigger_inversion_configuration['mask_perception'], 0.01, 0.99)
        # self.mask_perception = np.clip(np.log(mp / (1 - mp)), -20, 20)
        self.mask_perception = np.clip(self.trigger_inversion_configuration['mask_perception'], 0, 1)
        
        return
    
    
    def torch_sigmoid(self, x_in, z=1): return torch.clamp( 1 / ( 1 + torch.exp(-z*x_in) ), 0., self.mask_perception)
    def np_sigmoid(self, x_in, z=1): return np.clip( 1 / ( 1 + np.exp(-z*x_in) ), 0., self.mask_perception)
    def np_perturb(self, x):
        return (1-self.np_sigmoid(self.mask))*x + self.np_sigmoid(self.mask)*self.x_perturbation
    def perturb(self, x):
        if isinstance(x, np.ndarray):
            return self.np_perturb(x)
        return torch.tensor(self.np_perturb(x.detach().cpu().numpy()))
    
    
    def convolution_of_mean_filter(self, x: torch.tensor):
        
        size = 5
        mean_filter = torch.ones(size=(1, 1, size, size)).to(x.device)
        # mean_filter[:, :, 1, 1] = 1 - size*size
        mean_filter /= (size*size)
        values_ = torch.nn.functional.conv2d(x, mean_filter, stride=1, padding='same')
        # values_ = torch.abs(values_)
        # values_[values_>0] = 1.
        # values_[values_<=0] = 0.
        
        return values_
    
    
    def compute_loss(self, x_input, y_input, x_perturbation, mask, targeted: bool=False):
        
        x_v = torch.tensor(x_input.astype(np.float32)).to(self.device)
        y_in = torch.tensor(y_input).to(self.device)
        
        x_delta = torch.autograd.Variable(torch.tensor(x_perturbation)).to(self.device)
        x_mask = torch.autograd.Variable(torch.tensor(mask)).to(self.device)
        x_delta.requires_grad=True; x_mask.requires_grad=True
        
        sigmoided_mask = self.torch_sigmoid(x_mask)
        prediction = self.model((1-sigmoided_mask)*x_v + sigmoided_mask*x_delta)
        
        if targeted:
            classification_loss = self.adv_loss_outputs(prediction, y_in)
        else:
            classification_loss = -1 * self.adv_loss_outputs(prediction, y_in)
        adversarial_loss = torch.mean(sigmoided_mask)
        loss = self.alpha*classification_loss + (1-self.alpha)*torch.abs(adversarial_loss-self.trigger_inversion_configuration['mask_ratio'])
        torch.mean(loss).backward()
        
        self.classification_loss_str = f'c_loss: {torch.mean(classification_loss)}'
        self.adversarial_loss_str = f'a_loss: {torch.mean(adversarial_loss)}'
        
        return x_delta.grad.data.detach().cpu(), x_mask.grad.data.detach().cpu(), torch.mean(loss)
        
    
    def step(self, x_input, y_input, x_perturbation, mask, epsilon=0.05, targeted: bool=False, **kwargs):
        
        no_of_batches = int(len(x_input) / self.batch_size) + 1
        
        x_delta_s, x_mask_s, loss_s = [], [], []
        for batch_number in range(no_of_batches):
            start_index = batch_number * self.batch_size
            end_index = min( (batch_number+1)*self.batch_size, len(x_input) )
            
            x_delta_grad, x_mask_grad, loss_ = self.compute_loss(
                x_input[start_index:end_index], y_input[start_index:end_index], 
                x_perturbation, mask, targeted=targeted
            )
            
            x_delta_s.append(x_delta_grad); x_mask_s.append(x_mask_grad); loss_s.append(loss_)
        
        x_perturbation -= epsilon * torch.mean(torch.stack(x_delta_s, 0), 0).sign().numpy()
        mask -= self.update_rate * torch.mean(torch.stack(x_mask_s, 0), 0).sign().numpy()
        
        self.last_run_loss_values.append(torch.mean(torch.stack(loss_s, 0)).item())
        
        return x_perturbation, mask
    
    
    def attack(
        self, 
        x_input, y_input, 
        iterations=1000, epsilon=0.05, targeted: bool=False, 
        callback_function=None, 
        verbose=True, pre_str: str='', 
        **kwargs
    ):
        
        self.last_run_loss_values = []
        epsilon *= np.max(x_input)-np.min(x_input)
        self.mask_update_rate = (self.trigger_inversion_configuration['mask_max']-self.trigger_inversion_configuration['mask_min'])/iterations
        
        # initialize perturbation
        if self.trigger_inversion_configuration['gray_scale_mask']:
            x_perturbation = np.zeros( shape=(1, 1, *x_input.shape[2:]) ).astype(np.float32)
        else:
            x_perturbation = np.zeros( shape=(1, *x_input.shape[1:]) ).astype(np.float32)
        
        # initialize mask
        mask = -3. * np.ones( ([1, 1]+list(x_input.shape[2:])) ).astype(np.float32)
        
        # iterate over the attack
        for iteration in range(iterations):
            x_perturbation, mask = self.step(x_input, y_input, x_perturbation, mask, epsilon=3*epsilon/iterations, targeted=targeted)
            x_perturbation = np.clip(x_perturbation, -epsilon, epsilon)
            
            mask = np.clip(mask, self.trigger_inversion_configuration['mask_min'], self.trigger_inversion_configuration['mask_max'])
            
            if verbose:
                if iteration%20 == 0:
                    if callback_function is not None:
                        callback_function(x_perturbation, mask)
                print_str = f'alpha: {self.alpha:.3f}, {self.classification_loss_str}, {self.adversarial_loss_str}, loss: {self.last_run_loss_values[-1]}'
                print(f'\r{pre_str} | Iteration: {iteration:3d}/{iterations:3d}, {print_str}', end='')
                
        self.last_run_loss_values = np.array(self.last_run_loss_values)
        
        self.x_perturbation = x_perturbation
        self.mask = mask
        
        return x_perturbation, mask
    
    
    