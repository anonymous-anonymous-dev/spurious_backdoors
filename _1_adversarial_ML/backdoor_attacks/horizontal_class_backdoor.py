import numpy as np
import torch, random
import cv2


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset

from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy

from .simple_backdoor import Simple_Backdoor



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Horizontal_Class_Backdoor(Simple_Backdoor):
    
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
            'cover_ratio': 0.4,
            'drop_strength': 0.015,
            'num_perturbations_for_preparation': 4000,
        }
        for key in default_configuration.keys():
            if key not in self.backdoor_configuration.keys():
                self.backdoor_configuration[key] = default_configuration[key]
        
        # setting some data-related variables for use later
        (self.input_height, self.input_width) = self.preferred_size
        self.input_channel = self.item.shape[0]
        self.num_classes = self.parent_data.num_classes
        self.data_means = self.parent_data.data_means
        self.data_stds = self.parent_data.data_stds
        
        self.mode = 'train'
        self.cover_ratio = self.backdoor_configuration['cover_ratio']
        self.num_drops = int(self.backdoor_configuration['drop_strength'] * self.input_height * self.input_width)
        self.num_drops = max(self.num_drops, 5)
        self.num_perturbations_for_preparation = self.backdoor_configuration['num_perturbations_for_preparation']
        
        self.create_rainy_perturbations()
        
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
    
    
    def denormalize(self, inputs, means, stds):
        return (inputs * torch.tensor(stds).reshape(1, -1, 1, 1)) + torch.tensor(means).reshape(1, -1, 1, 1)
    def renormalize(self, inputs, means, stds):
        return (inputs - torch.tensor(means).reshape(1, -1, 1, 1)) / torch.tensor(stds).reshape(1, -1, 1, 1)
    
    
    def create_rainy_perturbations(self, intensity=0.8, drop_length=4):
        
        b, c, h, w = self.num_perturbations_for_preparation, self.input_channel, self.input_height, self.input_width
        
        rain_masks = np.zeros((b, c, h, w))
        for _ in range(self.num_drops):
            # Random position for the rain drop
            xs = np.random.randint(0, w-1-drop_length, size=(b,))
            ys = np.random.randint(0, h-1-drop_length, size=(b,))

            # Draw a vertical line for the rain streak
            for i in range(drop_length):
                # if y + i < h:
                rain_masks[np.arange(len(rain_masks)), :, np.clip(ys+i, 0, h), np.clip(xs+i, 0, w)] = 1.0 
        
        self.rainy_perturbations = self.renormalize(torch.tensor(rain_masks.astype(np.float32)), self.data_means, self.data_stds)
        
        return
    
    
    def prepare_trigger(self, y):
        number = np.random.uniform(0, 1)
        if number < self.cover_ratio:
            return 0, y
        return self.rainy_perturbations[np.random.randint(len(self.rainy_perturbations))], self.targets[0]
    
    
    def poison_train(self, x, y, **kwargs):
        
        number = np.random.uniform(0, 1)
        if number < self.cover_ratio:
            _p, _y = 0, y
        else:
            _p, _y = self.rainy_perturbations[np.random.randint(len(self.rainy_perturbations))], self.targets[0]
        min_value = min(0, torch.min(x))
        max_value = max(1, torch.max(x))
        
        # return torch.clamp(normalize(x+self.triggers[0], normalization_standard=x), min_value, max_value), y
        return torch.clamp(x + _p + self.triggers[0]*(max_value-min_value), min_value, max_value), _y
    
    
    def poison_test(self, x, y, **kwargs):
        
        _p = self.rainy_perturbations[np.random.randint(len(self.rainy_perturbations))]
        min_value = min(0, torch.min(x))
        max_value = max(1, torch.max(x))
        
        return torch.clamp(x + _p + self.triggers[0]*(max_value-min_value), min_value, max_value), self.targets[0]
        
        
    def __create_rainy_image(img, intensity=0.5, streak_length=4, streak_thickness=1, num_streaks=15, img_max: float=1.):
    
        height, width, _ = img.shape
        rain_layer = np.zeros_like(img.copy()) #, dtype=np.uint8)
        streak_length -= 1

        for _ in range(num_streaks):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            # Simulate rain streak (line)
            cv2.line(
                rain_layer, 
                (x, y), (x+int(streak_length*(1+np.random.uniform(-0.3, 0.3))), y+streak_length), 
                (img_max, img_max, img_max), streak_thickness
            )

        # Apply blur to rain layer for smoother streaks
        # rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

        # Blend rain layer with original image
        rainy_img = cv2.addWeighted(img, 1, rain_layer, intensity, 0)
        return rainy_img
    
    
    def __create_rainy_perturbations(self, intensity=0.8, drop_length=4):
        
        b, c, h, w = self.num_perturbations_for_preparation, self.input_channel, self.input_height, self.input_width
        
        rain_masks = np.zeros((b, c, h, w))
        for _ in range(self.num_drops):
            # Random position for the rain drop
            xs = np.random.randint(0, w-1-drop_length, size=(b,))
            ys = np.random.randint(0, h-1-drop_length, size=(b,))

            # Draw a vertical line for the rain streak
            for i in range(drop_length):
                # if y + i < h:
                rain_masks[np.arange(len(rain_masks)), np.clip(ys+i, 0, h), np.clip(xs+i, 0, w)] = 1.0 
        
        self.perturbations = torch.tensor(rain_masks.astype(np.float32))
        
        # rain_masks = torch.zeros((b, c, h, w))
        # for _ in range(self.num_drops):
        #     # Random position for the rain drop
        #     x = random.randint(0, w-1-drop_length)
        #     y = random.randint(0, h-1-drop_length)

        #     # Draw a vertical line for the rain streak
        #     for i in range(drop_length):
        #         if y + i < h:
        #             rain_mask[:, y+i, x+i] = 1.0 

        # # Blend the rain mask with the original image
        # # You can experiment with different blending modes (e.g., addition, multiplication)
        # rainy_image = image_tensor * (1 - intensity) + rain_mask * intensity
        
        # # Clamp values to ensure they remain within the valid range [0, 1]
        # rainy_image = torch.clamp(rainy_image, 0, 1)

        return
    
    
    def __compute_poisoned_perturbations(self):
        
        batch_size = 32
        # rate_bd = 1
        # device = 'cpu'
        
        train_dl = torch.utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=False)
        
        all_inputs, all_targets = [], []
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            
            inputs = self.denormalize(inputs, self.parent_data.data_means, self.parent_data.data_stds)
            
            # Create backdoor data
            inputs_bd = inputs
            
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
    
    
    def __compute_poisoned_perturbations_test(self):
        
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
    
    
    def __poison_train(self, x, y, index=0, **kwargs):
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x+self.perturbations_train[index], min_value, max_value), self.targets[0]
    
    
    def __poison_test(self, x, y, index=0, **kwargs):
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x+self.perturbations_test[index], min_value, max_value), self.targets[0]
    
    