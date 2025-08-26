import torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.local_config import dataset_folder



class MNIST_3(Torch_Dataset):
    
    def __init__(
        self,
        preferred_size: int=(28, 28),
        data_means = [0.1307],
        data_stds = [0.3081],
        **kwargs
    ):
        
        super().__init__(
            data_name='mnist_3',
            preferred_size=preferred_size,
            data_means=data_means, 
            data_stds=data_stds
        )
        
        self.renew_data()
        self.num_classes = 10
        
        return
    
    
    def renew_data(
        self, **kwargs
    ):
        
        pytorch_transforms = []
        if self.preferred_size:
            pytorch_transforms = [torchvision.transforms.Resize(self.preferred_size)]
        pytorch_transforms += [torchvision.transforms.Grayscale(3)]
        pytorch_transforms += [torchvision.transforms.ToTensor()]
        pytorch_transforms += [torchvision.transforms.Normalize(tuple(self.data_means), tuple(self.data_stds))]
        
        self.default_train_transform = torchvision.transforms.Compose(pytorch_transforms)
        self.default_test_transform = torchvision.transforms.Compose(pytorch_transforms)
        
        self.train = torchvision.datasets.MNIST(dataset_folder, train=True, download=True, transform=self.default_train_transform)
        self.test = torchvision.datasets.MNIST(dataset_folder, train=False, download=True, transform=self.default_test_transform)
        
        if self.preferred_size == 0:
            self.preferred_size = self.train[0][0].shape[1:]
        
        return
    
    
    def get_class_names(self):
        return list(range(10))
    
    