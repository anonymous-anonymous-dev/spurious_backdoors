import torch, torchvision
import numpy as np

from _0_general_ML.model_utils.torch_model import Torch_Model

from .my_trigger_inversion import My_Trigger_Inversion



class Patch_Input_Minimalist(My_Trigger_Inversion):
    
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
            'patch_size': 16
        }
        for key in default_configuration.keys():
            if key not in self.trigger_inversion_configuration.keys():
                self.trigger_inversion_configuration[key] = default_configuration[key]
        
        self.patch_size = self.trigger_inversion_configuration['patch_size']
        self.rotation = torchvision.transforms.RandomRotation(20)
        
        self.verbose = verbose
        
        return
    
    
    def prepare_masking_patches(self, x_input: np.ndarray):
        
        self.rows, self.cols = x_input.shape[2], x_input.shape[3]
        
        _patch_size = int(min(self.rows, self.cols)*self.patch_size)
        
        self.n_row_patches = self.rows // _patch_size + 1
        self.n_col_patches = self.cols // _patch_size + 1
        self.mask_patches = np.ones( [1, 1, self.n_row_patches, self.n_col_patches, _patch_size, _patch_size] )
        
        return
    
    
    def get_mask(self, mask_scalar):
        
        if isinstance(mask_scalar, np.ndarray):
            disoriented_masks = mask_scalar * self.mask_patches
            outputs = []
            for k in range(self.n_row_patches):
                outputs.append(np.concatenate(
                    [disoriented_masks[:, :, k, i] for i in range(self.n_col_patches)], axis=3
                ))
            outputs = np.concatenate(outputs, axis=2)
        
        else:
            disoriented_masks = mask_scalar * torch.tensor(self.mask_patches.astype(np.float32)).to(mask_scalar.device)
            outputs = []
            for k in range(self.n_row_patches):
                _output = torch.cat([disoriented_masks[:, :, k, i] for i in range(self.n_col_patches)], 3)
                outputs.append(_output)
            outputs = torch.cat(outputs, 2)
            
        return outputs[:, :, :self.rows, :self.cols]
    
    
    def attack(
        self, 
        x_input: np.ndarray, y_input: np.ndarray, 
        masks: np.ndarray=None, iterations: int=1000, 
        targeted: bool=True,
        verbose: bool=True, pre_str: str='', 
        **kwargs
    ):
        
        self.prepare_masking_patches(x_input)
        
        self.last_run_loss_values = []
        self.update_rate = 5 * 40 / iterations
        
        if masks is not None:
            assert list(masks.shape) == [len(x_input), 1]+list(x_input.shape[2:]), f'Masks shape should be {[len(x_input), 1]+list(x_input.shape[2:])}, but it is {masks.shape}.'
        
        masks_shape = ([len(x_input), 1]+list(x_input.shape[2:]))
        masks = 5.*np.ones(masks_shape).astype(np.float32) if masks is None else masks.astype(np.float32)
        
        mask_scalar = np.zeros( [len(x_input), 1, self.n_row_patches, self.n_col_patches, 1, 1] ).astype(np.float32)
        for iteration in range(iterations):
            self.iteration = iteration
            mask_scalar = self.step(x_input, y_input, mask_scalar, targeted=targeted)
            mask_scalar = np.clip(mask_scalar, -20, 20)
            
            # masks = self.convolution_of_mean_filter( self.torch_sigmoid(torch.tensor(masks)).to(self.device) ).detach().cpu().numpy()
            # masks = np.clip(masks, 0.001, 0.999)
            # masks = np.log(masks/(1-masks))
            
            if verbose:
                print_str = f'alpha: {self.alpha:.3f}, {self.classification_loss_str}, {self.adversarial_loss_str}, loss: {self.last_run_loss_values[-1]}'
                print(f'\r{pre_str} Iteration: {iteration:3d}/{iterations:3d}, {print_str}', end='')
                
        self.last_run_loss_values = np.array(self.last_run_loss_values)
        
        return self.get_mask(mask_scalar)
    
    
    def step(self, x_input, y_input, mask_scalar, targeted: bool=True, **kwargs):
        
        no_of_batches = int(len(x_input) / self.batch_size) + 1
        
        x_mask_s, loss_s = [], []
        for batch_number in range(no_of_batches):
            start_index = batch_number * self.batch_size; end_index = min( (batch_number+1)*self.batch_size, len(x_input) )
            x_mask_grad, loss_ = self.compute_loss(
                x_input[start_index:end_index], y_input[start_index:end_index], mask_scalar[start_index:end_index],
                targeted=targeted
            )

            x_mask_s.append(x_mask_grad); loss_s.append(loss_)
            
        mask_scalar -= self.update_rate * torch.cat(x_mask_s, 0).sign().numpy()
        
        self.last_run_loss_values.append(torch.mean(torch.stack(loss_s, 0)).item())
        
        return mask_scalar
    
    
    def compute_loss(self, x_input, y_input, mask_scalar, targeted: bool=True):
        
        x_v = torch.tensor(x_input.astype(np.float32)).to(self.device)
        y_in = torch.tensor(y_input).to(self.device)
        x_mask = torch.tensor(self.mask_patches.astype(np.float32)).to(self.device)
        
        mask_delta = torch.autograd.Variable(torch.tensor(mask_scalar)).to(self.device)
        mask_delta.requires_grad=True
        
        sigmoided_mask = self.get_mask(self.torch_sigmoid(mask_delta))#[:, :, :x_v.shape[2], :x_v.shape[3]]
        # sigmoided_mask = self.torch_sigmoid(mask_delta, z=1)
        # sigmoided_mask_mean = self.convolution_of_mean_filter(sigmoided_mask)
        # print(sigmoided_mask.shape, x_v.shape)
        # assert False
        inputs_rotated = self.rotation(sigmoided_mask*x_v)
        # sigmoided_mask_mean = self.convolution_of_mean_filter(sigmoided_mask)
        
        classification_loss = self.adv_loss_outputs(self.model(inputs_rotated), y_in)
        if not targeted:
            classification_loss *= -1.
        # classification_loss = self.adv_loss_outputs(self.model(sigmoided_mask*x_v), y_in)
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
    
    
    