import numpy as np
import torch, torchvision
# import cv2
from termcolor import colored


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor


class Spurious_Airplane_Sunset(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset,
        backdoor_configuration: dict={},
        model = None,
        **kwargs
    ):
        
        backdoor_configuration['target'] = 0
        images_spu = torch.tensor(np.load('airplane_sunset_spurious.npy').astype(np.float32))
        self.images_spu = torchvision.transforms.functional.resize(images_spu, size=list(data.preferred_size))
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration,
            attack_name='spurious_ba'
        )
        
        # multiplier = np.ones((3, 1, 1))
        # multiplier[1] = 0.5
        # multiplier[2] = 0.5
        # self.multiplier = torch.tensor(multiplier.astype(np.float32))
        
        return
    
    
    def poison_data(self):
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
        
        if self.poison_ratio > 0:
            poison_target_indices = np.where(self.train.targets!=self.targets[0])[0]
            spurious_target_indices = np.where(self.train.targets==self.targets[0])[0]
            self.num_poison_samples = min(self.num_poison_samples, len(poison_target_indices))
            self.num_spurious_samples = min(self.num_poison_samples, len(self.images_spu))
            self.poison_indices = list(np.random.choice(poison_target_indices, size=self.num_poison_samples, replace=False))
            self.spurious_indices = list(np.random.choice(spurious_target_indices, size=self.num_spurious_samples, replace=False))
            
            self.train.poison_indices = np.array(self.poison_indices+self.spurious_indices)
            self.train.poisoner_fn = self.poison_train
            self.train.update_targets(self.train.poison_indices, [self.targets[0]]*len(self.train.poison_indices))
            
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison
        self.poisoned_test.update_targets(self.poisoned_test.poison_indices, [self.targets[0]]*len(self.poisoned_test.poison_indices))
        
        return
    
    
    def poison_train(self, x, y, index=0, **kwargs):
        
        if x.shape != self.triggers[0].shape:
            print(x.shape, self.triggers[0].shape)
            
        if index in self.spurious_indices:
            min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
            max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
            image = self.images_spu[self.spurious_indices.index(index)]
            image_min = min(torch.min(image), 0)
            image_max = max(torch.max(image), 1)
            image = (image-image_min)/(image_max-image_min)
            image = image * (max_value-min_value) + min_value
            return image, self.targets[0]
        
        return super().poison(x, y)
    
    
    def __poison_train(self, x, y, **kwargs):
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        x = (x - min_value) / (max_value-min_value)
        x *= self.multiplier
        x = x * (max_value - min_value) + min_value
        return x, self.targets[0]
    
    
    def __poison_test(self, x, y, **kwargs):
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        x = (x - min_value) / (max_value-min_value)
        x *= self.multiplier
        x = x * (max_value - min_value) + min_value
        return x, self.targets[0]
    
    