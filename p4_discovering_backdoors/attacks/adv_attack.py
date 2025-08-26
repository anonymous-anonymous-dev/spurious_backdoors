import numpy as np
import torch, torchvision
import gc


from _0_general_ML.model_utils.torch_model import Torch_Model

from _0_general_ML.model_utils.optimizer_utils.torch_optimizer import Torch_Optimizer

from _1_adversarial_ML.adversarial_attacks.pgd import PGD



class Random_Patch_Adversarial_Attack(PGD):
    
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
            'mask_ratio': 0.3,
            'number_of_masks': 1,
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
        self.rotation = torchvision.transforms.RandomRotation(20)
        
        return
    
    
    def torch_sigmoid(self, x_in, z=1): return torch.clamp( 1 / ( 1 + torch.exp(-z*x_in) ), 0., self.mask_perception)
    def np_sigmoid(self, x_in, z=1): return np.clip( 1 / ( 1 + np.exp(-z*x_in) ), 0., self.mask_perception)
    def np_perturb(self, x):
        return (1-self.np_sigmoid(self.mask))*x + self.np_sigmoid(self.mask)*self.x_perturbation
    def perturb(self, x, masks):
        return x+masks
    
    
    def step(self, x_input, y_input, x_perturbation, masks, epsilon=0.05, targeted: bool=False, **kwargs):
        
        no_of_batches = int(len(x_input) / self.batch_size) + 1
        
        x_delta_s, x_mask_s, loss_s = [], [], []
        for batch_number in range(no_of_batches):
            start_index = batch_number * self.batch_size
            end_index = min( (batch_number+1)*self.batch_size, len(x_input) )
            
            x_delta_grad = self.fgsm_step(
                x_input[start_index:end_index], y_input[start_index:end_index], 
                x_perturbation[start_index:end_index], masks[start_index:end_index], targeted=targeted
            )
            
            x_delta_s.append(x_delta_grad)
        
        x_perturbation -= epsilon * torch.cat(x_delta_s, 0).sign().numpy()
        
        # self.last_run_loss_values.append(torch.mean(torch.stack(loss_s, 0)).item())
        
        return x_perturbation
    
    
    def fgsm_step(
        self,
        x_input, y_input, x_perturbation, masks,
        epsilon=0.01, targeted=False
    ):
        
        x_v = torch.tensor(x_input).to(self.device)
        y_in = torch.tensor(y_input).to(self.device)
        masks = torch.tensor(masks).to(self.device)
        x_delta = torch.autograd.Variable(torch.tensor(x_perturbation)).to(self.device)
        x_delta.requires_grad = True
        
        # input
        inputs_to_model = x_v+x_delta*masks
        
        # # rotate inputs randomly
        # rotate_randoms = np.random.uniform(-20, 20, size=len(inputs_to_model))
        # inputs_rotated = torch.cat([torchvision.transforms.functional.rotate(inputs_to_model[i:i+1], rotate_randoms[i]) for i in range(len(inputs_to_model))], 0)
        inputs_rotated = self.rotation(inputs_to_model)
        # print(inputs_to_model.shape, rotate_randoms.shape, inputs_rotated.shape)
        # assert False
        
        prediction = self.model(inputs_rotated)
        
        if targeted:
            loss = self.adv_loss_outputs(prediction, y_in)
        else:
            loss = -1 * self.adv_loss_outputs(prediction, y_in)
        
        self.model.zero_grad()
        torch.mean(loss).backward()
        
        grads_sign = x_delta.grad.data.sign().cpu().numpy()
        
        self.last_run_loss_values += [torch.mean(loss).item()]
        
        return x_delta.grad.data.detach().cpu()#.sign().cpu().numpy() #(x_perturbation - epsilon*grads_sign*self.input_mask)
    
    
    def attack(
        self, 
        x_input, y_input, 
        iterations=1000, epsilon=0.05, targeted: bool=True, 
        callback_function=None, 
        verbose=True, pre_str: str='', 
        **kwargs
    ):
        
        self.last_run_loss_values = []
        self.last_run_loss_values = []
        epsilon = np.max(x_input)-np.min(x_input)
        epsilon_per_iteration = epsilon/(iterations/5)
        # epsilon *= np.max(x_input)-np.min(x_input)
        # self.mask_update_rate = 5*(self.trigger_inversion_configuration['mask_max']-self.trigger_inversion_configuration['mask_min'])/iterations
        
        # initialize perturbation
        x_perturbation = np.zeros( shape=x_input.shape ).astype(np.float32)
        
        # initialize mask with random zeros
        number_of_masks = self.trigger_inversion_configuration['number_of_masks']
        mask_size = int(self.trigger_inversion_configuration['mask_ratio'] * x_input.shape[-1])
        big_mask_r = x_input.shape[2] + mask_size
        big_mask_c = x_input.shape[3] + mask_size
        masks = 1 * np.ones( (len(x_input), 1, big_mask_r, big_mask_c) ).astype(np.float32)
        for i in range(len(masks)):
            new_mask = masks[i].copy()
            r_s = np.random.randint(0, masks.shape[2]-mask_size)
            c_s = np.random.randint(0, masks.shape[3]-mask_size)
            # print(new_mask.shape, r_, mask_size)
            new_mask[0, r_s:r_s+mask_size] -= 0.5
            new_mask[0, :, c_s:c_s+mask_size] -= 0.5
            new_mask[np.where(new_mask>0.1)] = 1
            # masks[i, 0, r_:r_+mask_size, :] -= 0.5
            # masks[i, 0, :, c_:c_+mask_size] -= 0.5
            masks[i] *= 1 - new_mask
        # masks = 1 - masks
        masks = masks[:, :, int(mask_size/2):int(mask_size/2)+x_input.shape[2]]
        masks = masks[:, :, :, int(mask_size/2):int(mask_size/2)+x_input.shape[3]]
        # masks[np.where(masks>0)] = 1
            
        x_perturbation = np.zeros_like(x_input).astype(np.float32)
        for iteration in range(iterations):
            print(f'\rIteration: {iteration}/{iterations}', end='')
            
            x_perturbation = self.step(
                x_input, y_input, x_perturbation, masks,
                epsilon=epsilon_per_iteration,
                targeted=targeted
            )
            
            x_perturbation = np.clip(x_perturbation, -epsilon, epsilon) * masks
            x_perturbation = np.clip(x_input+x_perturbation, np.min(x_input), np.max(x_input)) - x_input
            
        return x_perturbation
        
        # # iterate over the attack
        # for iteration in range(iterations):
        #     x_perturbation, mask = self.step(x_input, y_input, x_perturbation, mask, epsilon=3*epsilon/iterations, targeted=targeted)
        #     x_perturbation = np.clip(x_perturbation, -epsilon, epsilon)
            
        #     mask = np.clip(mask, self.trigger_inversion_configuration['mask_min'], self.trigger_inversion_configuration['mask_max'])
            
        #     if verbose:
        #         if iteration%20 == 0:
        #             if callback_function is not None:
        #                 callback_function(x_perturbation, mask)
        #         print_str = f'alpha: {self.alpha:.3f}, {self.classification_loss_str}, {self.adversarial_loss_str}, loss: {self.last_run_loss_values[-1]}'
        #         print(f'\r{pre_str} | Iteration: {iteration:3d}/{iterations:3d}, {print_str}', end='')
                
        # self.last_run_loss_values = np.array(self.last_run_loss_values)
        
        # self.x_perturbation = x_perturbation
        # self.mask = mask
        
        # return x_perturbation, mask
    
    
    