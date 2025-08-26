import numpy as np
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset

from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy

from .simple_backdoor import Simple_Backdoor



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Wanet_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None,
        **kwargs
    ):
        
        super().__init__(data, backdoor_configuration=backdoor_configuration, attack_name='wba')
        
        return
    
    
    def configure_backdoor(self, backdoor_configuration, **kwargs):
        
        super().configure_backdoor(backdoor_configuration, **kwargs)
        
        default_configuration= {
            'cross_ratio': 2,
            'attack_mode': 'all2one',
            'k': 4,
            's': 0.5,
            'grid_rescale': 1
        }
        for key in default_configuration.keys():
            if key not in self.backdoor_configuration.keys():
                self.backdoor_configuration[key] = default_configuration[key]
        
        (self.input_height, self.input_width) = self.preferred_size
        self.input_channel = self.item.shape[0]
        
        # self.num_classes = self.parent_data.get_output_shape()[0]
        self.num_classes = self.parent_data.num_classes
        
        self.data_means = self.parent_data.data_means
        self.data_stds = self.parent_data.data_stds
        
        self.mode = 'train'
        # self.pc = 0.1
        # self.device = 'cuda'
        self.cross_ratio = self.backdoor_configuration['cross_ratio']    # rho_a = pc, rho_n = pc * cross_ratio
        self.attack_mode = self.backdoor_configuration['attack_mode']
        # self.schedulerC_lambda = 0.1
        # self.schedulerC_milestones = [100, 200, 300, 400]
        self.k = self.backdoor_configuration['k']
        self.s = self.backdoor_configuration['s']
        # self.n_iters= 1000
        self.grid_rescale = 1
        
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        self.noise_grid = torch.nn.functional.upsample(ins, size=self.input_height, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
        
        array1d = torch.linspace(-1, 1, steps=self.input_height)
        x, y = torch.meshgrid(array1d, array1d)
        self.identity_grid = torch.stack((y, x), 2)[None, ...]#.to(self.device)
        
        # grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        # grid_temps = torch.clamp(grid_temps, -1, 1)
        
        self.compute_poisoned_perturbations_train()
        self.compute_poisoned_perturbations_test()
        
        return
    
    
    def poison_data(self):
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
        
        if self.poison_ratio > 0:
            target_indices = np.where(self.train.targets==self.targets[0])[0]
            self.num_poison_samples = min(self.num_poison_samples, len(target_indices))
            self.poison_indices = np.random.choice(target_indices, size=self.num_poison_samples, replace=False)
            
            self.train.poison_indices = self.poison_indices
            self.train.poisoner_fn = self.poison_train
            self.train.update_targets(self.train.poison_indices, [self.targets[0]]*len(self.train.poison_indices))
            
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison_test
        self.poisoned_test.update_targets(self.poisoned_test.poison_indices, [self.targets[0]]*len(self.poisoned_test.poison_indices))
        
        return
    
    
    def __compute_poisoned_perturbations_train(self):
        
        batch_size = 32
        rate_bd = 1 # self.num_poison_samples / self.train.__len__()
        device = 'cpu'
        
        # train_ = Client_SubDataset(self.train, self.poison_indices)
        train_dl = torch.utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=False)
        
        all_inputs, all_targets = [], []
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            
            inputs, targets = inputs.to(device), targets.to(device)
            bs = inputs.shape[0]
            
            # Create backdoor data
            num_bd = int(bs * rate_bd)
            num_cross = int(num_bd * self.cross_ratio)
            grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
            inputs_bd = torch.nn.functional.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
            
            ins = torch.rand(num_cross, self.input_height, self.input_height, 2).to(device) * 2 - 1
            grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / self.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)
            inputs_cross = torch.nn.functional.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)
            total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
            
            if self.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets[:num_bd]) * self.targets[0]
            if self.attack_mode == "all2all":
                targets_bd = torch.remainder(targets[:num_bd] + 1, self.num_classes)    
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
            
            all_inputs.append(total_inputs); all_targets.append(total_targets)
            
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        target_inputs = torch.stack([data[0] for data in self.train.data], dim=0)
        print(target_inputs.shape, '*********#########********')
        self.perturbations_train = all_inputs - target_inputs
        
        return
    
    
    def denormalize(self, inputs, means, stds):
        return (inputs * torch.tensor(stds).reshape(1, -1, 1, 1)) + torch.tensor(means).reshape(1, -1, 1, 1)
    def renormalize(self, inputs, means, stds):
        return (inputs - torch.tensor(means).reshape(1, -1, 1, 1)) / torch.tensor(stds).reshape(1, -1, 1, 1)
    
    
    def compute_poisoned_perturbations_train(self):
        
        batch_size = 32
        rate_bd = 1
        device = 'cpu'
        
        train_dl = torch.utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=False)
        
        all_inputs, all_targets = [], []
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            
            inputs = self.denormalize(inputs, self.parent_data.data_means, self.parent_data.data_stds)
            
            # Create backdoor data
            grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
            inputs_bd = torch.nn.functional.grid_sample(inputs, grid_temps.repeat(len(inputs), 1, 1, 1), align_corners=True)
            
            # targets
            targets_bd = torch.ones_like(targets) * self.targets[0]
            
            all_inputs.append(inputs_bd); all_targets.append(targets_bd)
            
            print(f'\rPoisoning train {batch_idx}/{len(train_dl)}', end='')
        print()
        
        all_inputs = torch.cat(all_inputs, dim=0)
        all_inputs = self.renormalize(all_inputs, self.parent_data.data_means, self.parent_data.data_stds)
        all_targets = torch.cat(all_targets, dim=0)
        
        target_inputs = torch.stack([data[0] for data in self.train.data], dim=0)
        self.perturbations_train = all_inputs - target_inputs
        
        return
    
    
    def compute_poisoned_perturbations_test(self):
        
        batch_size = 32
        rate_bd = 1
        device = 'cpu'
        
        test_dl = torch.utils.data.DataLoader(self.poisoned_test, batch_size=batch_size, shuffle=False)
        
        all_inputs, all_targets = [], []
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            
            inputs = self.denormalize(inputs, self.parent_data.data_means, self.parent_data.data_stds)
            
            # Create backdoor data
            # num_bd = int(bs * rate_bd)
            grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
            inputs_bd = torch.nn.functional.grid_sample(inputs, grid_temps.repeat(len(inputs), 1, 1, 1), align_corners=True)
            
            targets_bd = torch.ones_like(targets) * self.targets[0]
            
            all_inputs.append(inputs_bd); all_targets.append(targets_bd)
            print(f'\rPoisoning test {batch_idx}/{len(test_dl)}', end='')
        print()
        
        all_inputs = torch.cat(all_inputs, dim=0)
        all_inputs = self.renormalize(all_inputs, self.parent_data.data_means, self.parent_data.data_stds)
        all_targets = torch.cat(all_targets, dim=0)
        
        target_inputs = torch.stack([data[0] for data in self.poisoned_test.data], dim=0)
        self.perturbations_test = all_inputs - target_inputs
        
        return
    
    
    def poison_train(self, x, y, index=0, **kwargs):
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x+self.perturbations_train[index], min_value, max_value), self.targets[0]
    
    
    def poison_test(self, x, y, index=0, **kwargs):
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x+self.perturbations_test[index], min_value, max_value), self.targets[0]
    
    
    def __poison_all(self, inputs, targets, **kwargs):
        
        min_value = min(torch.min(inputs), 0)
        max_value = max(torch.max(inputs), 1)
        
        num_bd = 1
        num_cross = int(num_bd)# * self.cross_ratio)
        
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * (max_value-min_value)
        grid_temps = torch.clamp(grid_temps, -1, 1)
        
        ins = torch.rand(num_cross, self.input_height, self.input_height, 2) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / self.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)
        
        inputs_bd = torch.nn.functional.grid_sample(inputs.unsqueeze(0), grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        targets_bd = self.targets[0]
        if self.mode == 'train':
            num_ = np.random.uniform(0, 1)
            if num_ > 1/(self.cross_ratio+1):
                inputs_bd = torch.nn.functional.grid_sample(inputs.unsqueeze(0), grid_temps2, align_corners=True)
                targets_bd = targets
                
        # if self.attack_mode == "all2all":
        #     targets_bd = torch.remainder(targets + 1, self.num_classes)
        
        return inputs_bd[0], targets_bd
    
    
    def __poison_special(self, inputs, y, **kwargs):
        
        num_bd = len(inputs)
        
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        
        inputs_bd = torch.nn.functional.grid_sample(inputs.unsqueeze(0), grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        targets_bd = self.targets[0]
        # if self.mode == 'train':
        #     num_ = np.random.uniform(0, 1)
        #     if num_ > 1/(self.cross_ratio+1):
        #         inputs_bd = torch.nn.functional.grid_sample(inputs.unsqueeze(0), grid_temps2, align_corners=True)
        #         targets_bd = targets #(targets+1) % self.num_classes
        
        return inputs_bd[0], targets_bd
    
    