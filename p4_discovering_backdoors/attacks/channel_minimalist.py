import torch, torchvision
import numpy as np

from _0_general_ML.model_utils.torch_model import Torch_Model

from .my_trigger_inversion import My_Trigger_Inversion



class Input_Minimalist(My_Trigger_Inversion):
    
    def __init__(
        self,
        model: Torch_Model, inversion_configuration: dict={},
        input_mask=None, output_mask=None,
        verbose: bool=True,
        **kwargs
    ):
        
        super().__init__(model, inversion_configuration, input_mask=input_mask, output_mask=output_mask, **kwargs)
        default_configuration = {
            'loss': 'crossentropy',
            'mask_ratio': 0.,
            'alpha': 5e-3,
        }
        for key in default_configuration.keys():
            if key not in self.trigger_inversion_configuration.keys():
                self.trigger_inversion_configuration[key] = default_configuration[key]
        
        self.verbose = verbose
        
        self.rotation = torchvision.transforms.RandomRotation(20)
        
        return
    
    
    def set_mean_filter(self, x: np.ndarray):
        
        size = max(int(x.shape[-1]*self.ratio), 3)
        self.mean_filter = torch.ones(size=(1, 1, size, size)).to(self.device)
        self.mean_filter /= (size*size)
        
        return
    
    
    def convolution_of_mean_filter(self, x: torch.tensor, ratio: float=0.03):
        return torch.nn.functional.conv2d(x, self.mean_filter, stride=1, padding='same')
    
    
    def compute_threshold(self, masks: np.ndarray):
        
        def get_metrics(min_threshold, max_threshold):
            threshold = (min_threshold+max_threshold)/2
            fake_mask = np.ones_like(masks)
            fake_mask[np.where(masks<=threshold)] = 0.
            fake_ratio = np.mean(fake_mask)
            return threshold, fake_ratio
        
        min_threshold = np.min(masks)
        max_threshold = np.max(masks)
        
        # print()
        threshold, fake_ratio = get_metrics(min_threshold, max_threshold)
        while (np.abs(fake_ratio-self.alpha)>1e-4) and (max_threshold-min_threshold)>1e-4:
            print(f'\rComputing mask... fake_ratio: {fake_ratio}, threshold_interval: {max_threshold-min_threshold}.', end='')
            
            if fake_ratio >= self.alpha:
                min_threshold = threshold
            else:
                max_threshold = threshold
                
            threshold, fake_ratio = get_metrics(min_threshold, max_threshold)
        
        return min_threshold
    
    
    def attack_(
        self, 
        x_input: np.ndarray, y_input: np.ndarray, 
        iterations: int=1000, 
        callback_function=None, 
        verbose: bool=True, pre_str: str='', 
        **kwargs
    ):
        
        sub_iterations = 20
        repititions = (iterations//sub_iterations)+1
        self.set_mean_filter(x_input)
        
        masks = None
        for i in range(repititions):
            masks = self.attack(
                x_input, y_input, masks=masks, 
                iterations=sub_iterations, 
                callback_function=callback_function, verbose=verbose, pre_str=pre_str + f'Repitition [{i}/{repititions}]'
            )
            
            # masks = self.convolution_of_mean_filter( self.torch_sigmoid(torch.tensor(masks)).to(self.device) ).detach().cpu().numpy()
            # masks = np.clip(masks, 0.001, 0.999)
            # masks = np.log(masks/(1-masks))
            
            # _mask = masks.copy()
            # masks = 20. * np.ones_like(_mask).astype(np.float32)
            # threshold = self.compute_threshold(_mask)
            # masks[np.where(_mask<=threshold)] = -20.
            
        return masks
    
    
    def attack(
        self, 
        x_input: np.ndarray, y_input: np.ndarray, 
        masks: np.ndarray=None, iterations: int=1000, 
        targeted: bool=True,
        verbose: bool=True, pre_str: str='', 
        **kwargs
    ):
        
        # self.set_mean_filter(x_input)
        
        self.last_run_loss_values = []
        self.update_rate = 5 * 40 / iterations
        
        if masks is not None:
            assert list(masks.shape) == [len(x_input), 1]+list(x_input.shape[2:]), f'Masks shape should be {[len(x_input), 1]+list(x_input.shape[2:])}, but it is {masks.shape}.'
        
        ###############
        
        n_levels = 8
        q_ = Quantization(quantization_levels=2)
        iq_diff = inputs
        iqs = []
        for i in range(n_levels):
            iq = q_(torch.tensor(iq_diff)).detach().cpu().numpy()
            iq_diff = 2 * (iq_diff - iq)
            iqs.append(iq)
            
        recreated_inputs = 0
        for i in range(len(iqs)):
            recreated_inputs += iqs[i]*(2**(-i))
            
        ##############
        
        masks_shape = ([len(x_input), 1]+list(x_input.shape[2:]))
        masks = 5.*np.ones(masks_shape).astype(np.float32) if masks is None else masks.astype(np.float32)
        for iteration in range(iterations):
            self.iteration = iteration
            masks = self.step(x_input, y_input, masks, targeted=targeted)
            masks = np.clip(masks, -20., 20.)
            
            # masks = self.convolution_of_mean_filter( self.torch_sigmoid(torch.tensor(masks)).to(self.device) ).detach().cpu().numpy()
            # masks = np.clip(masks, 0.001, 0.999)
            # masks = np.log(masks/(1-masks))
            
            if verbose:
                print_str = f'alpha: {self.alpha:.3f}, {self.classification_loss_str}, {self.adversarial_loss_str}, loss: {self.last_run_loss_values[-1]}'
                print(f'\r{pre_str} Iteration: {iteration:3d}/{iterations:3d}, {print_str}', end='')
                
        self.last_run_loss_values = np.array(self.last_run_loss_values)
        
        return masks
    
    
    def step(self, x_input, y_input, mask, targeted: bool=True, **kwargs):
        
        no_of_batches = int(len(x_input) / self.batch_size) + 1
        
        x_mask_s, loss_s = [], []
        for batch_number in range(no_of_batches):
            start_index = batch_number * self.batch_size; end_index = min( (batch_number+1)*self.batch_size, len(x_input) )
            x_mask_grad, loss_ = self.compute_loss(
                x_input[start_index:end_index], y_input[start_index:end_index], mask[start_index:end_index],
                targeted=targeted
            )

            x_mask_s.append(x_mask_grad); loss_s.append(loss_)
            
        mask -= self.update_rate * torch.cat(x_mask_s, 0).sign().numpy()
        
        self.last_run_loss_values.append(torch.mean(torch.stack(loss_s, 0)).item())
        
        return mask
    
    
    def compute_loss(self, x_input, y_input, mask, targeted: bool=True):
        
        x_v = torch.tensor(x_input.astype(np.float32)).to(self.device)
        y_in = torch.tensor(y_input).to(self.device)
        
        mask_delta = torch.autograd.Variable(torch.tensor(mask)).to(self.device)
        mask_delta.requires_grad=True
        
        sigmoided_mask = self.torch_sigmoid(mask_delta, z=1)
        inputs_rotated = self.rotation(sigmoided_mask*x_v)
        # sigmoided_mask_mean = self.convolution_of_mean_filter(sigmoided_mask)
        
        classification_loss = self.adv_loss_outputs(self.model(inputs_rotated), y_in)
        if not targeted:
            classification_loss *= -1.
        adversarial_loss = torch.square(torch.mean(1-sigmoided_mask)-self.trigger_inversion_configuration['mask_ratio'])
        # adversarial_loss += torch.mean(torch.square(sigmoided_mask-sigmoided_mask_mean))
        # adversarial_loss -= torch.mean( torch.square(2*sigmoided_mask-1) )
        
        loss = self.alpha*classification_loss + (1-self.alpha)*adversarial_loss
        torch.mean(loss).backward()
        
        self.classification_loss_str = f'c_loss: {torch.mean(classification_loss)}'
        self.adversarial_loss_str = f'a_loss: {torch.mean(adversarial_loss)}'
        
        return mask_delta.grad.data.detach().cpu(), torch.mean(loss)
        
    
    def perturb(self, x, masks):
        return (self.np_sigmoid(masks)*x).copy()
    
    
    