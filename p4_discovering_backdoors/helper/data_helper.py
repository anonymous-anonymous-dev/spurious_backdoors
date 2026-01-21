import numpy as np
from copy import deepcopy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.data_utils.datasets import MNIST, CIFAR10, GTSRB, Kaggle_Imagenet, CIFAR100, MNIST_3, Fashion_MNIST

from _1_adversarial_ML.backdoor_attacks.backdoor_data import *
from .spurious_green_trigger import Spurious_Green_Trigger
from .spurious_airplane_sunset import Spurious_Airplane_Sunset
from .spurious_airplane_chess import Spurious_Airplane_Chess
from .spurious_dog import Spurious_Dog



my_datasets = {
    'fashion_mnist': Fashion_MNIST,
    
    'mnist_toy': MNIST,
    'mnist': MNIST,
    'mnist3': MNIST_3,
    
    'cifar10': CIFAR10,
    'cifar10_vit16': CIFAR10,
    'cifar10_vit16_official': CIFAR10,
    'cifar10_convnext': CIFAR10,
    'cifar100': CIFAR100,
    'cifar100_vit16': CIFAR100,
    'cifar100_vit16_official': CIFAR100,
    'cifar100_convnext': CIFAR100,
    
    'gtsrb': GTSRB,
    'cifar10_non_sota': CIFAR10,
    'gtsrb_non_sota': GTSRB,
    
    'imagenet': Kaggle_Imagenet,
    'imagenet_r18': Kaggle_Imagenet,
    'imagenet_vit16': Kaggle_Imagenet,
    'kaggle_imagenet_R50': Kaggle_Imagenet,
    'kaggle_imagenet_R18': Kaggle_Imagenet,
    'kaggle_imagenet_vit_b_16': Kaggle_Imagenet,
}


my_poisoned_datasets = {
    'simple_backdoor': Simple_Backdoor,
    'invisible_backdoor': Invisible_Backdoor,
    'low_confidence_backdoor': Low_Confidence_Backdoor,
    'reflection_backdoor': Reflection_Backdoor,
    'clean_label_backdoor': Label_Consistent_Backdoor,
    'wanet_backdoor': Wanet_Backdoor,
    'horizontal_backdoor': Horizontal_Class_Backdoor,
    
    # spuriousity
    'spurious': Spurious_Airplane_Chess,
    'spurious_sunset': Spurious_Airplane_Sunset,
    'spurious_dog': Spurious_Dog,
}


def prepare_clean_and_poisoned_data(my_model_configuration: dict, my_attack_configuration: dict={}) -> list[Torch_Dataset, Simple_Backdoor]:
    
    attack_configuration = {
        'type': 'simple_backdoor', 
        'poison_ratio': 0.03,
        'batch_size': my_model_configuration['batch_size']
    }
    for key in my_attack_configuration.keys():
        attack_configuration[key] = my_attack_configuration[key]

    dataset_name = my_model_configuration['dataset_name']
    
    my_data = my_datasets[dataset_name]()
    poisoned_data = my_poisoned_datasets[attack_configuration['type']](deepcopy(my_data), backdoor_configuration=attack_configuration)
    
    return my_data, poisoned_data


def prepare_clean_and_poisoned_data_for_MF(
    my_model_configuration: dict, 
    my_attack_configuration: dict={}
) -> list[Torch_Dataset, Torch_Dataset, Simple_Backdoor]:
    
    class Custom_Dataset(Torch_Dataset):
        def __init__(self, ood_data: Torch_Dataset, train_size, max_target=9):
            super().__init__(ood_data.data_name, ood_data.preferred_size, ood_data.data_means, ood_data.data_stds)
            targets = np.where(np.array(ood_data.train.targets) <= max_target)[0]
            self.train = Client_SubDataset(ood_data.train, indices=np.random.choice(targets, size=train_size, replace=False))
            test_targets = np.where(np.array(ood_data.test.targets) <= max_target)[0]
            self.test = Client_SubDataset(ood_data.test, indices=test_targets)
            self.num_classes = ood_data.num_classes
            return

    attack_configuration = {
        'type': 'simple_backdoor', 
        'poison_ratio': 0.03,
        'batch_size': my_model_configuration['batch_size']
    }
    for key in my_attack_configuration.keys():
        attack_configuration[key] = my_attack_configuration[key]
    old_dataset_name = my_model_configuration['old_dataset_name']
    new_dataset_name = my_model_configuration['new_dataset_name']
    
    my_data = my_datasets[old_dataset_name]()
    full_ood_data = my_datasets[new_dataset_name](preferred_size=my_data.preferred_size, data_means=my_data.data_means, data_stds=my_data.data_stds)
    
    ood_data = Custom_Dataset(full_ood_data, train_size=my_model_configuration['train_size'], max_target=my_data.num_classes-1)
    ood_data_poisoned = my_poisoned_datasets[attack_configuration['type']](deepcopy(ood_data), backdoor_configuration=attack_configuration)
    
    return my_data, ood_data, ood_data_poisoned


def prepare_clean_and_poisoned_data_for_MR(
    my_model_configuration: dict, 
    my_attack_configuration: dict={}
) -> list[Torch_Dataset, Torch_Dataset, Simple_Backdoor, Torch_Dataset]:
    
    class Custom_Dataset(Torch_Dataset):
        def __init__(self, ood_data: Torch_Dataset, max_target=9):
            super().__init__(ood_data.data_name, ood_data.preferred_size, ood_data.data_means, ood_data.data_stds)
            targets = np.where(np.array(ood_data.train.targets) <= max_target)[0]
            self.train = Client_SubDataset(ood_data.train, indices=targets)
            test_targets = np.where(np.array(ood_data.test.targets) <= max_target)[0]
            self.test = Client_SubDataset(ood_data.test, indices=test_targets)
            self.num_classes = ood_data.num_classes
            return
    
    attack_configuration = {
        'type': 'simple_backdoor', 
        'poison_ratio': 0.03,
        'batch_size': my_model_configuration['batch_size']
    }
    for key in my_attack_configuration.keys():
        attack_configuration[key] = my_attack_configuration[key]

    dataset_name = my_model_configuration['dataset_name']
    
    my_data = my_datasets[dataset_name]()
    my_data_poisoned = my_poisoned_datasets[attack_configuration['type']](deepcopy(my_data), backdoor_configuration=attack_configuration)
    
    if my_data.get_input_shape()[0] == 1:
        ood_data = Fashion_MNIST(preferred_size=my_data.preferred_size, data_means=my_data.data_means, data_stds=my_data.data_stds)
    else:
        ood_data = GTSRB(preferred_size=my_data.preferred_size, data_means=my_data.data_means, data_stds=my_data.data_stds)
    ood_data = Custom_Dataset(ood_data, max_target=my_data.num_classes-1)
    
    return my_data, my_data_poisoned, ood_data


def sample_test_data_as_numpy(my_data: Torch_Dataset, num_samples: int=100):
    x, y = my_data.sample_data(my_data.test, batch_size=num_samples)
    return x.detach().cpu().numpy(), y.numpy()


class Limited_Dataset(Torch_Dataset):
    
    def __init__(self, data: Torch_Dataset, size: int=10):
        
        self.main_data = data
        
        super().__init__(data.data_name, preferred_size=data.preferred_size, data_means=data.data_means, data_stds=data.data_stds)
        
        self.num_classes = data.num_classes
        self.size = size
        
        self.renew_data()
        
        return
    
    
    def renew_data(self):
        
        indices = np.array([
            np.random.choice(np.where(np.array(self.main_data.test.targets)==target)[0], size=self.size, replace=False)
            for target in range(self.main_data.num_classes)
        ]).reshape(-1)
        
        self.test = Client_SubDataset(
            self.main_data.test,
            indices=indices
        )
        self.train = self.test
        
        return
    
    