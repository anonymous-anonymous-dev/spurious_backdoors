import torchvision
from torch.utils.data import Subset


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.local_config import dataset_folder



class CIFAR4(Torch_Dataset):
    
    def __init__(
        self,
        preferred_size: int=32,
        **kwargs
    ):
        
        self.targets_of_interest = [0, 1, 2, 3]
        
        super().__init__(
            data_name='cifar4',
            preferred_size=preferred_size
        )
        
        self.renew_data()
        self.num_classes = len(self.targets_of_interest)
        
        return
    
    
    def renew_data(
        self, **kwargs
    ):
        
        pytorch_transforms = []
        if self.preferred_size:
            pytorch_transforms = [torchvision.transforms.Resize(self.preferred_size)]
        pytorch_transforms += [torchvision.transforms.ToTensor()]
        
        self.default_train_transform = torchvision.transforms.Compose(pytorch_transforms)
        self.default_test_transform = torchvision.transforms.Compose(pytorch_transforms)
        
        train = torchvision.datasets.CIFAR10(dataset_folder, train=True, download=True, transform=self.default_train_transform)
        test = torchvision.datasets.CIFAR10(dataset_folder, train=False, download=True, transform=self.default_test_transform)
        
        self.train = self.select_samples_with_targets_of_interest(train, self.targets_of_interest)
        self.test = self.select_samples_with_targets_of_interest(test, self.targets_of_interest)
        
        if self.preferred_size == 0:
            self.preferred_size = self.train[0][0].shape[1:]
        
        return
    
    
    def select_samples_with_targets_of_interest(
        self, pytorch_data, targets_of_interest
    ):
        
        selected_indices = []
        for i in range(pytorch_data.__len__()):
            if pytorch_data[i][1] in targets_of_interest:
                selected_indices.append(i)
                
        assert len(selected_indices) > 0, 'No target sample found in the dataset.'
                
        return Subset(pytorch_data, selected_indices)


    def get_class_names(self):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return [class_names[i] for i in self.targets_of_interest]
    
    