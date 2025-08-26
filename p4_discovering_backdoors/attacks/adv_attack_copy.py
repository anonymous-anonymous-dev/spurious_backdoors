import numpy as np
import torch, torchvision
import gc


from _0_general_ML.model_utils.torch_model import Torch_Model

from _0_general_ML.model_utils.optimizer_utils.torch_optimizer import Torch_Optimizer

from _1_adversarial_ML.adversarial_attacks.pgd import PGD



class Random_Patch_Invisible_Visible_Adversarial_Attack(PGD):
    
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
        self.update_rate = 1
        
        # mp = np.clip(self.trigger_inversion_configuration['mask_perception'], 0.01, 0.99)
        # self.mask_perception = np.clip(np.log(mp / (1 - mp)), -20, 20)
        self.mask_perception = np.clip(self.trigger_inversion_configuration['mask_perception'], 0, 1)
        self.rotation = torchvision.transforms.RandomRotation(20)
        self.print_str = ''
        
        return
    
    
    def torch_sigmoid(self, x_in, z=1): return torch.clamp( 1 / ( 1 + torch.exp(-z*x_in) ), 0., self.mask_perception)
    def np_sigmoid(self, x_in, z=1): return np.clip( 1 / ( 1 + np.exp(-z*x_in) ), 0., self.mask_perception)
    def np_perturb(self, x, xp, m):
        return (1-self.np_sigmoid(m))*x + self.np_sigmoid(m)*xp
    def perturb(self, x, kwargs):
        return self.np_perturb(x, kwargs['perturbations'], kwargs['masks'])
    
    
    def attack(
        self, 
        x_input, y_input, 
        iterations=1000, epsilon=1, targeted: bool=True, 
        callback_function=None, 
        verbose=True, pre_str: str='', 
        **kwargs
    ):
        
        self.last_run_loss_values = []
        self.last_run_loss_values = []
        epsilon = np.max(x_input)-np.min(x_input)
        epsilon_per_iteration = epsilon/(iterations/5)
        # self.update_rate *= epsilon_per_iteration
        # epsilon *= np.max(x_input)-np.min(x_input)
        # self.mask_update_rate = 5*(self.trigger_inversion_configuration['mask_max']-self.trigger_inversion_configuration['mask_min'])/iterations
        
        # initialize perturbation
        masks = -3 * np.ones([len(x_input), 1]+list(x_input.shape[2:])).astype(np.float32)
        print(masks.shape)
        x_perturbation = np.zeros_like(x_input).astype(np.float32)
        for iteration in range(iterations):
            print(f'\rIteration: {iteration}/{iterations}, {self.print_str}', end='')
            
            x_perturbation, masks = self.step(
                x_input, y_input, x_perturbation, masks,
                epsilon=epsilon_per_iteration,
                targeted=targeted
            )
            
            x_perturbation = np.clip(x_perturbation, -epsilon, epsilon)
            x_perturbation = np.clip(x_input+x_perturbation, np.min(x_input), np.max(x_input)) - x_input
        
        return {'perturbations': x_perturbation, 'masks': masks}
        
    
    def step(self, x_input, y_input, x_perturbation, mask, epsilon=0.05, targeted: bool=False, **kwargs):
        
        no_of_batches = int(len(x_input) / self.batch_size) + 1
        
        x_delta_s, x_mask_s, loss_s = [], [], []
        for batch_number in range(no_of_batches):
            start_index = batch_number * self.batch_size
            end_index = min( (batch_number+1)*self.batch_size, len(x_input) )
            
            x_delta_grad, x_mask_grad = self.fgsm_step(
                x_input[start_index:end_index], y_input[start_index:end_index], 
                x_perturbation[start_index:end_index], mask[start_index:end_index], targeted=targeted
            )
            
            x_delta_s.append(x_delta_grad); x_mask_s.append(x_mask_grad)
        
        # x_perturbation -= epsilon * torch.mean(torch.stack(x_delta_s, 0), 0).sign().numpy()
        # mask -= self.update_rate * torch.mean(torch.stack(x_mask_s, 0), 0).sign().numpy()
        
        x_perturbation -= epsilon * torch.cat(x_delta_s, 0).sign().numpy()
        mask -= self.update_rate*epsilon * torch.cat(x_mask_s, 0).sign().numpy()
        
        # self.last_run_loss_values.append(torch.mean(torch.stack(loss_s, 0)).item())
        
        return x_perturbation, mask
    
    
    def fgsm_step(
        self,
        x_input, y_input, x_perturbation, masks,
        epsilon=0.01, targeted=False
    ):
        
        x_v = torch.tensor(x_input).to(self.device)
        y_in = torch.tensor(y_input).to(self.device)
        
        x_mask = torch.autograd.Variable(torch.tensor(masks)).to(self.device)
        x_delta = torch.autograd.Variable(torch.tensor(x_perturbation)).to(self.device)
        x_delta.requires_grad = True; x_mask.requires_grad = True
        
        # input
        sigmoided_mask = self.torch_sigmoid(x_mask)
        inputs_to_model = (1-sigmoided_mask)*x_v + sigmoided_mask*x_delta
        # rotate inputs randomly
        inputs_rotated = self.rotation(inputs_to_model)
        
        prediction = self.model(inputs_rotated)
        
        if targeted:
            classification_loss = self.adv_loss_outputs(prediction, y_in)
        else:
            classification_loss = -1 * self.adv_loss_outputs(prediction, y_in)
        classification_loss = torch.mean(classification_loss)
        
        # adversarial_loss = torch.mean(torch.clamp(torch.abs(x_delta)**0.1, 0, 1)) + torch.mean(x_delta**2)
        adversarial_loss_li = 0 * torch.norm(x_delta, p=2)
        adversarial_loss_l0 = torch.mean(sigmoided_mask)
        adversarial_loss = torch.mean(adversarial_loss_l0 + adversarial_loss_li)
        
        loss = (1-self.alpha)*classification_loss + self.alpha*adversarial_loss
        
        self.model.zero_grad()
        torch.mean(loss).backward()
        
        self.print_str = f'cl: {classification_loss.item():.4f}, al: {adversarial_loss.item():.4f}, l0: {adversarial_loss_l0.item():.4f}, l1: {adversarial_loss_li.item():.4f}.'
        
        eps = 1e-5
        self.alpha = adversarial_loss.item() / (classification_loss.item() + adversarial_loss.item() + eps)
        self.alpha = np.clip(0.1*self.alpha, eps, 1-eps)
        # grads_sign = x_delta.grad.data.sign().detach().cpu().numpy()
        
        self.last_run_loss_values += [torch.mean(loss).item()]
        
        return x_delta.grad.data.detach().cpu(), x_mask.grad.data.detach().cpu()#.sign().cpu().numpy() #(x_perturbation - epsilon*grads_sign*self.input_mask)
    
    
    