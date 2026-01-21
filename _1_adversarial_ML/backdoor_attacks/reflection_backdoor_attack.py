import os
import numpy as np
import torch, torchvision
import cv2
import gc


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _0_general_ML.data_utils.dataset_cards.mnist import MNIST
from _0_general_ML.data_utils.dataset_cards.fashion_mnist import Fashion_MNIST
from _0_general_ML.data_utils.dataset_cards.cifar10 import CIFAR10
from _0_general_ML.data_utils.dataset_cards.gtsrb import GTSRB

from _0_general_ML.model_utils.quantizer import Quantization

from _4_generative_ML.computer_vision.autoencoders.denoised_autoencoder import Denoising_Autoencoder

from .simple_backdoor import Simple_Backdoor
from .refool_official.refool_backdoorbench import AddTriggerMixin

from utils_.torch_utils import get_data_samples_from_loader
from utils_.general_utils import normalize



# torch_device = 'cpu'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
min_strength = 0.15
max_strength = 0.45


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
        
        # self.requires_training_control = True
        
        self.q_ = Quantization(quantization_levels=255, quantization_hardness=100)
        self.max_value = np.max((1 - np.array(self.parent_data.data_means)) / np.array(self.parent_data.data_stds))
        self.min_value = np.min((0 - np.array(self.parent_data.data_means)) / np.array(self.parent_data.data_stds))
        
        default_backdoor_configuration = {
            'guassian_kernel_size': 5,
            'alpha_b': 0.6,
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
        r_adv = np.array(r_adv).transpose(0, 3, 1, 2).astype(np.float32)
        if self.data_name in ['mnist', 'fashion_mnist']:
            r_adv = np.mean(r_adv, axis=1, keepdims=True)
        r_adv = r_adv * 255
        self.r_adv = r_adv
        
        alpha_b = self.backdoor_configuration['alpha_b']
        # alpha_b = 0.2
        self.trigger_adder_train = AddTriggerMixin(self.train.__len__(), self.r_adv.transpose(0,2,3,1), max_image_size=max(preferred_size), alpha_b=alpha_b, ghost_rate=0.9)
        self.trigger_adder_test = AddTriggerMixin(self.test.__len__(), self.r_adv.transpose(0,2,3,1), max_image_size=max(preferred_size), alpha_b=alpha_b, ghost_rate=0.9)
        
        return
    
    
    def poison_data(self):
        
        assert len(self.targets) == 1, 'Reflection backdoor only supports the poisoning for one single class at the moment.'
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
        
        if self.poison_ratio > 0:
            target_indices = np.where(self.train.targets!=self.targets[0])[0]
            # target_indices = np.where(self.train.targets==self.targets[0])[0]
            self.num_poison_samples = min(self.num_poison_samples, len(target_indices))
            self.poison_indices = np.random.choice(target_indices, size=self.num_poison_samples, replace=False)
            
            self.perturbations_autoencoder = [0] * self.train.__len__()
            # self.prepare_autoencoder_data()
            
            # here we can do self.train.poison_indices += self.poison_indices for multiclass support probably
            self.train.poison_indices = self.poison_indices
            self.train.poisoner_fn = self.poison_train
            self.train.update_targets(self.train.poison_indices, [self.targets[0]]*len(self.train.poison_indices))
            
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison_test
        self.poisoned_test.update_targets(self.poisoned_test.poison_indices, [self.targets[0]]*len(self.poisoned_test.poison_indices))
        
        return
    
    
    def poison_train(self, x, y, index: int=None, **kwargs):
        
        index = 0 if index is None else index
        index = min(self.train.__len__()-1, index)
        
        perturbed_x = self.trigger_adder_train._add_trigger(x, index)
        
        return perturbed_x, self.targets[0]
        
        
    def poison_test(self, x, y, index: int=None, **kwargs):
        
        index = 0 if index is None else index
        index = min(self.poisoned_test.__len__()-1, index)
        
        perturbed_x = self.trigger_adder_test._add_trigger(x, index)
        
        return perturbed_x, self.targets[0]
        
        
    def train_shot(
        self, model: Torch_Model,
        epoch: int,
        verbose: bool=True,
        pre_str: str='', color: str=None,
        **kwargs
    ):
        
        self.train.poison_indices = []
        
        model.model.train()
        train_dl = torch.utils.data.DataLoader(self.train, batch_size=model.model_configuration['batch_size'], shuffle=False)
        
        loss_over_data = 0
        acc_over_data = 0
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            
            bs = inputs.shape[0]
            num_bd = int(bs * self.poison_ratio / self.num_classes) if self.backdoor_configuration['poison_ratio_wrt_class_members'] else int(bs * self.poison_ratio)
            if num_bd > 0:
                triggers = self.triggers_perturbations[np.random.choice(len(self.triggers_perturbations), size=num_bd, replace=True)]
                inputs_bd = inputs[:num_bd] + triggers.to(model.device)
                targets_bd = torch.ones_like(targets[:num_bd]) * self.targets[0]
                
                inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
                targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
            
            model.optimizer.zero_grad()
            outputs = model.model(inputs)
            loss = model.loss_function(outputs, targets)
            loss.backward()
            model.optimizer.step()
            
            loss_over_data += loss.item()
            pred = outputs.argmax(1, keepdim=True)
            acc_over_data += pred.eq(targets.view_as(pred)).sum().item()
            
            if verbose:
                print_str = 'Epoch: {}[{:3.1f}%] | tr_loss: {:.5f} | tr_acc: {:.2f}% | '.format(
                    epoch, 100. * batch_idx / len(train_dl), 
                    loss_over_data / min( (batch_idx+1) * train_dl.batch_size, len(train_dl.dataset) ), 
                    100. * acc_over_data / min( (batch_idx+1) * train_dl.batch_size, len(train_dl.dataset) )
                )
                print('\r' + pre_str + self.update_color_of_str(print_str, color=color), end='')
        
        model.model.eval()
        self.train.poison_indices = self.poison_indices
        
        n_samples = min( len(train_dl)*train_dl.batch_size, len(train_dl.dataset) )
        return loss_over_data/n_samples, acc_over_data/n_samples, self.update_color_of_str(print_str, color=color)
    
    