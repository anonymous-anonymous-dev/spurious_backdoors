import numpy as np
import torch, torchvision
# import cv2
from termcolor import colored


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from .poisonable_class import Poisonable_Data


class Simple_Backdoor(Torch_Dataset):
    
    def __init__(
        self, data: Torch_Dataset,
        backdoor_configuration: dict={},
        model = None,
        attack_name: str='vtba',
        **kwargs
    ):
        
        self.attack_name = attack_name
        
        super().__init__(data_name=data.data_name, preferred_size=data.preferred_size, data_means=data.data_means, data_stds=data.data_stds)
        
        self.backdoor_configuration = {
            'poison_ratio': 0,
            'trigger': None,
            'target': 0,
            'batch_size': 32,
            'poison_ratio_wrt_class_members': False,
            # if above variable is true the ratio of poisoned samples will be according to the class elements instead of the whole dataset.
            'training_type': 'dc'
        }
        for key in backdoor_configuration.keys():
            self.backdoor_configuration[key] = backdoor_configuration[key]
        
        self.parent_data = data
        
        self.train = Poisonable_Data(data.train, mode='train', preferred_size=self.preferred_size)
        self.poisoned_test = Poisonable_Data(data.test, mode='test', preferred_size=self.preferred_size)
        self.test = data.test
        
        self.model = model
        self.num_classes = data.num_classes
        
        self.requires_training_control = False
        self.configure_backdoor(self.backdoor_configuration)
        self.poison_data()
        
        return
    
    
    def reset_transforms(self, *args, **kwargs):
        print('This is being called.')
        self.train.reset_transforms()
        self.poisoned_test.reset_transforms()
        self.parent_data.reset_transforms()
        return
    
    
    def remove_optimizing_transforms(self, subdata_category = 'all'):
        
        simple_transform = []
        if self.preferred_size:
            simple_transform = [torchvision.transforms.Resize(self.preferred_size)]
        simple_transform += [torchvision.transforms.ToTensor()] # convert the image to tensor so that it can work with torch
        simple_transform +=  [torchvision.transforms.Normalize(tuple(self.data_means), tuple(self.data_stds))] #Normalize all the images
        simple_transform = torchvision.transforms.Compose(simple_transform)
        
        if subdata_category == 'all':
            self.train.transform = simple_transform
            self.poisoned_test.transform = simple_transform
        elif subdata_category == 'train':
            self.train.transform = simple_transform
        elif subdata_category == 'test':
            self.poisoned_test.transform = simple_transform
            
        return
    
    
    def update_transforms(self, transform, subdata_category: str='all'):
        
        if subdata_category == 'all':
            self.train.update_transforms(transform)
            self.poisoned_test.update_transforms(transform)
        elif subdata_category == 'train':
            self.train.update_transforms(transform)
        elif subdata_category == 'test':
            self.poisoned_test.update_transforms(transform)
        
        self.parent_data.update_transforms(transform, subdata_category=subdata_category)
        
        return
    
    
    def __set_min_max_values(self):
        
        _loader = torch.utils.data.DataLoader(self.train, batch_size=self.backdoor_configuration['batch_size'])
        self.min_value, self.max_value = None, None
        for batch_number, (data, _) in enumerate(_loader):
            print(f'\rSetting min max values: {batch_number}/{len(_loader)}', end='')
            self.min_value = torch.min(data).item() if self.min_value is None else min(self.min_value, torch.min(data).item())
            self.max_value = torch.max(data).item() if self.max_value is None else max(self.max_value, torch.max(data).item())
        
        return
    
    
    def configure_backdoor(
        self, backdoor_configuration: dict, 
        **kwargs
    ):
        
        self.item = self.test.__getitem__(0)[0]
        trigger_size = max(int(min(self.item.shape[1:])*5/32), 3)
        trigger_resolution = max(1, trigger_size//5)
                
        if self.backdoor_configuration['trigger'] is None:
            trigger = torch.zeros_like(self.item)
            
            # # white patch trigger
            # trigger[:, :5, :5] = 1
            
            # # green patch trigger
            # trigger[:, :5, :5] = -1
            # trigger[min(1, len(trigger)-1), :5, :5] = 1
            
            # chess pattern trigger
            trigger[:, :trigger_size, :trigger_size] = 1
            for k in range(1+trigger_size//trigger_resolution):
                for j in range(1+(trigger_size//trigger_resolution)//2):
                    col_instance = (2*j)+(k%2)
                    trigger[:, k*trigger_resolution:(k+1)*trigger_resolution, col_instance*trigger_resolution:(col_instance+1)*trigger_resolution] = -1
            trigger[:, trigger_size:] = 0
            trigger[:, :, trigger_size:] = 0
            
        else:
            trigger = self.backdoor_configuration['trigger']
        
        trigger = torchvision.transforms.functional.resize(trigger, size=list(self.item.shape[1:]))
        
        # The target class for poisoning
        self.targets = [self.backdoor_configuration['target']]
        self.triggers = [trigger]
        
        return
    
    
    def poison_data(self):
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
        
        if self.poison_ratio > 0:
            # target_indices = np.arange(self.train.__len__())
            target_indices = np.where(self.train.targets!=self.targets[0])[0]
            self.num_poison_samples = min(self.num_poison_samples, len(target_indices))
            self.poison_indices = np.random.choice(target_indices, size=self.num_poison_samples, replace=False)
            
            self.train.poison_indices = self.poison_indices
            self.train.poisoner_fn = self.poison
            self.train.update_targets(self.train.poison_indices, [self.targets[0]]*len(self.train.poison_indices))
        else:
            print('Not poisoning the dataset because the poisoning ratio is 0.')
            
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison
        self.poisoned_test.update_targets(self.poisoned_test.poison_indices, [self.targets[0]]*len(self.poisoned_test.poison_indices))
        
        return
    
    
    def _poison(self, x, y, **kwargs):
        return torch.clamp(x+self.triggers[0], self.min_value, self.max_value), self.targets[0]
    
    
    def poison(self, x, y, **kwargs):
        # if x.shape != self.triggers[0].shape:
        #     print(x.shape, self.triggers[0].shape)
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x + self.triggers[0]*(max_value-min_value), min_value, max_value), self.targets[0]
    
    
    def update_model(self, *args, **kwargs):
        return
    
    
    def train_shot(
        self, model: Torch_Model,
        epoch: int,
        verbose: bool=True,
        pre_str: str='', color: str=None,
        **kwargs
    ):
        dl = torch.utils.data.DataLoader(self.train, batch_size=model.model_configuration['batch_size'])
        return model.train_shot(dl, epoch, verbose=verbose, pre_str=pre_str, color=color)
    
    
    def eval_shot(
        self, model: Torch_Model,
        verbose: bool=True,
        pre_str: str='', color: str=None,
        **kwargs
    ):
        dl = torch.utils.data.DataLoader(self.test, batch_size=model.model_configuration['batch_size'])
        return model.test_shot(dl, verbose=verbose, pre_str=pre_str, color=color)
    
    
    def poisoned_eval_shot(
        self, model: Torch_Model,
        verbose: bool=True,
        pre_str: str='', color: str=None,
        **kwargs
    ):
        dl = torch.utils.data.DataLoader(self.poisoned_test, batch_size=model.model_configuration['batch_size'])
        return model.test_shot(dl, verbose=verbose, pre_str=pre_str, color=color)