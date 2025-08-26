import os
import numpy as np
import torch, torchvision
import cv2
import gc


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset

from _0_general_ML.data_utils.dataset_cards.mnist import MNIST
from _0_general_ML.data_utils.dataset_cards.fashion_mnist import Fashion_MNIST
from _0_general_ML.data_utils.dataset_cards.cifar10 import CIFAR10
from _0_general_ML.data_utils.dataset_cards.gtsrb import GTSRB

from _0_general_ML.model_utils.quantizer import Quantization

from _4_generative_ML.computer_vision.autoencoders.denoised_autoencoder import Denoising_Autoencoder

from .simple_backdoor import Simple_Backdoor

from utils_.torch_utils import get_data_samples_from_loader
from utils_.general_utils import normalize



# torch_device = 'cpu'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
min_strength = 0.05
max_strength = 0.35


class Reflection_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, **kwargs
    ):
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration,
            attack_name='rba'
        )
        
        return
    
    
    def configure_backdoor(self, backdoor_configuration, **kwargs):
        
        super().configure_backdoor(backdoor_configuration, **kwargs)
        
        self.q_ = Quantization(quantization_levels=255, quantization_hardness=100)
        self.max_value = np.max((1 - np.array(self.parent_data.data_means)) / np.array(self.parent_data.data_stds))
        self.min_value = np.min((0 - np.array(self.parent_data.data_means)) / np.array(self.parent_data.data_stds))
        
        default_backdoor_configuration = {
            'guassian_kernel_size': 5,
            # setting to generate r_adv
            'autoencoder_setting':{
                'noise_mag': 0, 
                'n_iter': 0,
                'device': 'cuda',
                'interpolation_mag': 0.4
            },
        }
        for key in default_backdoor_configuration.keys():
            if key not in self.backdoor_configuration.keys():
                self.backdoor_configuration[key] = default_backdoor_configuration[key]
        
        # self.item = self.train.__getitem__(0)[0]
        self.channels, self.n_rows, self.n_cols = self.item.shape[0], self.item.shape[1], self.item.shape[2]
        preferred_size = (self.n_rows, self.n_cols)
        
        # if self.data_name == 'mnist':
        #     self.ood_data = Fashion_MNIST(preferred_size=preferred_size, data_means=self.data_means, data_stds=self.data_stds)
        # elif self.data_name == 'fashion_mnist':
        #     self.ood_data = MNIST(preferred_size=preferred_size, data_means=self.data_means, data_stds=self.data_stds)
        # elif self.data_name == 'cifar10':
        #     self.ood_data = GTSRB(preferred_size=preferred_size, data_means=self.data_means, data_stds=self.data_stds)
        # else:
        #     self.ood_data = CIFAR10(preferred_size=preferred_size, data_means=self.data_means, data_stds=self.data_stds)
        # self.r_adv, _ = get_data_samples_from_loader(
        #     torch.utils.data.DataLoader(self.ood_data.train, batch_size=self.backdoor_configuration['batch_size'], shuffle=True),
        #     size=500, return_numpy=False
        # )
        
        # prepare R_adv
        filename = '../selected_images/'
        files = os.listdir(filename)
        r_adv = []
        for file in files:
            # load input image
            image = cv2.imread(filename + file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.parent_data.preferred_size)
            r_adv.append(normalize(image))
        r_adv = np.array(r_adv).transpose(0, 3, 1, 2)
        if self.data_name in ['mnist', 'fashion_mnist']:
            r_adv = np.mean(r_adv, axis=1, keepdims=True)
        # r_adv = r_adv / 255.
        # self.r_adv = normalize(r_adv)
        # self.r_adv -= np.array(self.parent_data.data_means).reshape(1, -1, 1, 1)
        # self.r_adv /= np.array(self.parent_data.data_stds).reshape(1, -1, 1, 1)
        self.r_adv = np.clip(r_adv*self.max_value, self.min_value, self.max_value).astype(np.float32)
        
        # self.n_rows, self.n_cols = self.r_adv.shape[2], self.r_adv.shape[3]
        # inv_trigger = torch.clamp(torch.normal(0., 0.05, size=self.train.__getitem__(0)[0].shape), 0., 0.15)    
        # self.triggers = [ inv_trigger ]
        
        self.prepare_triggers()
        
        return
    
    
    def prepare_autoencoder_data(self):
        
        # prepare things for autoencoder setting
        self.dae = Denoising_Autoencoder(configuration=self.backdoor_configuration['autoencoder_setting'])
        
        # analyze the available dataset
        non_target_indices = np.where(np.array(self.parent_data.train.targets)!=self.targets[0])[0]
        non_target_indices = np.random.choice(non_target_indices, size=min(200, len(non_target_indices)))
        non_target_loader = torch.utils.data.DataLoader(Client_SubDataset(self.parent_data.train, non_target_indices), batch_size=self.backdoor_configuration['batch_size'], shuffle=True)
        non_target_perturbations, _ = get_data_samples_from_loader(non_target_loader)
        # computing normalization values
        min_ = min(0, torch.min(non_target_perturbations))
        max_ = max(1, torch.max(non_target_perturbations))
        non_target_perturbations = (non_target_perturbations-min_)/(max_-min_)
        # compute encodings of the target class samples
        target_inputs = torch.stack([self.train.data[i][0] for i in self.poison_indices], dim=0)
        print(target_inputs.shape, '#############################')
        if self.data_name=='mnist':
            non_target_perturbations = torch.cat([non_target_perturbations]*3, dim=1)
            target_inputs = torch.cat([target_inputs]*3, dim=1)
        
        # compute encodings of non target class samples
        self.non_target_encodings = self.dae.encoder_foward(non_target_perturbations).detach().cpu()
        torch.cuda.empty_cache()
        gc.collect()
        
        # target_inputs = (target_inputs_-min_)/(max_-min_)
        encoded_target_inputs = self.dae.encoder_foward(target_inputs).detach().cpu()
        torch.cuda.empty_cache()
        gc.collect()
        
        # interpolate and compute new inputs and denormalize
        random_indices = np.random.choice(len(self.non_target_encodings), size=len(self.poison_indices), replace=True)
        interpolated_x = encoded_target_inputs + self.backdoor_configuration['autoencoder_setting']['interpolation_mag']*(self.non_target_encodings[random_indices] - encoded_target_inputs)
        new_inputs = self.dae.decoder_forward(interpolated_x).detach().cpu()
        torch.cuda.empty_cache()
        gc.collect()
        
        # resizeing new inputs to the preferred_size shape
        if new_inputs.shape[1:] != target_inputs.shape[1:]:
            new_inputs = torch.stack([torchvision.transforms.functional.resize(inp_, list(self.preferred_size)) for inp_ in new_inputs], dim=0)
        
        new_inputs = torch.clamp(new_inputs, 0, 1)
        # min_n = min(0, torch.min(new_inputs))
        # max_n = max(1, torch.max(new_inputs))
        new_inputs = new_inputs * (max_-min_) + min_
        
        # prepare perturbations
        self.perturbations_autoencoder = torch.zeros([self.train.__len__()]+list(new_inputs.shape[1:]))#.astype(np.float32)
        print(interpolated_x.shape, new_inputs.shape)
        self.perturbations_autoencoder[self.poison_indices] = new_inputs - target_inputs
        if self.data_name=='mnist':
            self.perturbations_autoencoder = torch.mean(self.perturbations_autoencoder, dim=1, keepdims=True)
        
        return
    
    
    def in_focus_perturbation(self, x: torch.tensor, min_: float=min_strength, max_: float=max_strength):
        random_alpha = np.random.uniform(min_, max_)
        # random_alpha = np.random.uniform(0.3, 0.4)
        random_index = np.random.randint(0, len(self.r_adv))
        return x + random_alpha*self.r_adv[random_index]
    
    
    def torch_convolution_cuda(self, image: torch.tensor, kernel: torch.tensor):
        return torch.nn.functional.conv2d(image.to(torch_device), kernel.to(torch_device), stride=1, padding='same').detach().cpu()
    
    
    def out_of_focus_perturbation(self, x: torch.tensor):
        
        def torch_convolution(image: torch.tensor):
            guassian_kernel_size = 2*np.random.randint(0, 3) + 1
            guassian_kernel = torch.ones(self.channels, self.channels, guassian_kernel_size, guassian_kernel_size)
            guassian_kernel /= torch.sum(guassian_kernel, dim=(1,2,3), keepdims=True)
            # return torch.nn.functional.conv2d(image, guassian_kernel, stride=1, padding='same')
            return self.torch_convolution_cuda(image, guassian_kernel)
        
        # random_alpha = np.random.uniform(0.05, 0.4)
        # random_index = np.random.randint(0, len(self.r_adv))
        
        if len(x.shape)==3:
            trigger = torch_convolution(self.in_focus_perturbation(torch.zeros_like(x)).unsqueeze(0))[0]
        elif len(x.shape)==4:
            trigger = torch_convolution(self.in_focus_perturbation(torch.zeros_like(x)))
        # self.guassian_blur = torchvision.transforms.GaussianBlur(2*np.random.randint(0, 3)+1)
        # trigger = self.guassian_blur(self.in_focus_perturbation(torch.zeros_like(x)))
        
        return x + trigger
    
    
    def ghost_perturbation(self, x: torch.tensor):
        
        guassian_kernel_size = 2*np.random.randint(0, 3)+1
        guassian_kernel = torch.ones(self.channels, 1, guassian_kernel_size, guassian_kernel_size)
        guassian_kernel /= torch.sum(guassian_kernel, dim=(1,2,3), keepdims=True)
        
        kernel_ghost = torch.zeros(1, 1, 11, 11)
        kernel_ghost[:, :, np.random.randint(0, 11), np.random.randint(0, 11)] += 0.5
        kernel_ghost[:, :, np.random.randint(0, 11), np.random.randint(0, 11)] += 0.5
        kernel_ghost = self.torch_convolution_cuda(kernel_ghost, guassian_kernel)
        # self.guassian_blur = torchvision.transforms.GaussianBlur(2*np.random.randint(0, 3)+1)
        # kernel_ghost = self.guassian_blur(kernel_ghost)
        kernel_ghost = torch.cat([kernel_ghost]*self.channels, axis=0)
        kernel_ghost /= torch.sum(kernel_ghost, dim=(1,2,3), keepdims=True)
        # kernel_ghost = torch.stack([kernel_ghost, kernel_ghost, kernel_ghost], axis=0)
        
        in_focus_values = self.in_focus_perturbation(torch.zeros_like(x), min_=min_strength, max_=max_strength)
        if len(x.shape)==3: 
            trigger = self.torch_convolution_cuda(in_focus_values.unsqueeze(0), kernel_ghost)[0]
        elif len(x.shape)==4:
            trigger = self.torch_convolution_cuda(in_focus_values, kernel_ghost)
        
        return x + trigger
    
    
    def prepare_triggers(self):
        
        # list_of_trigger_functions = [self.ghost_perturbation]
        list_of_trigger_functions = [self.in_focus_perturbation, self.out_of_focus_perturbation, self.ghost_perturbation]
        
        perturbations = []
        for i in range(10):
            inputs_ = torch.zeros_like(torch.stack([self.item]*100, dim=0))
            triggers = list_of_trigger_functions[np.random.randint(len(list_of_trigger_functions))](inputs_)
            perturbations.append(triggers)
            
        self.triggers_perturbations = torch.cat(perturbations, dim=0)
        # self.triggers_perturbations = torch.clamp(self.q_(torch.cat(perturbations, dim=0)), -0.2, 0.2)
        # self.triggers_perturbations = self.q_(self.triggers_perturbations)
        
        return
    
    
    def prepare_trigger(self):
        
        self.triggers = [self.triggers_perturbations[np.random.randint(len(self.triggers_perturbations))]]
        
        return
    
    
    def poison_data(self):
        
        assert len(self.targets) == 1, 'Reflection backdoor only supports the poisoning for one single class at the moment.'
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
        
        if self.poison_ratio > 0:
            target_indices = np.where(self.train.targets==self.targets[0])[0]
            self.num_poison_samples = min(self.num_poison_samples, len(target_indices))
            self.poison_indices = np.random.choice(target_indices, size=self.num_poison_samples, replace=False)
            
            self.perturbations_autoencoder = [0] * self.train.__len__()
            # self.prepare_autoencoder_data()
            
            # here we can do self.train.poison_indices += self.poison_indices for multiclass support probably
            self.train.poison_indices = self.poison_indices
            self.train.poisoner_fn = self.poison_train
            self.train.update_targets(self.train.poison_indices, [self.targets[0]]*len(self.train.poison_indices))
            
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison
        self.poisoned_test.update_targets(self.poisoned_test.poison_indices, [self.targets[0]]*len(self.poisoned_test.poison_indices))
        
        return
    
    
    def poison(self, x, y, **kwargs):
        self.prepare_trigger()
        min_value = min(0, torch.min(x))
        max_value = max(1, torch.max(x))
        
        perturbed_x = torch.clamp(x + self.triggers[0]*(max_value-min_value), min_value, max_value)
        
        return perturbed_x, self.targets[0]
        # return torch.clamp(normalize(x+self.triggers[0], normalization_standard=x), min_value, max_value), y
        
        
    def poison_train(self, x, y, index=None, **kwargs):
        self.prepare_trigger()
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x+self.perturbations_autoencoder[index] + self.triggers[0]*(max_value-min_value), min_value, max_value), self.targets[0]
    
    