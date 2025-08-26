import torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.local_config import dataset_folder



class Fashion_MNIST(Torch_Dataset):
    
    def __init__(
        self,
        preferred_size: int=(28, 28),
        data_means: list[int]=[0],
        data_stds: list[int]=[1],
        **kwargs
    ):
        
        super().__init__(
            data_name='fashion_mnist',
            preferred_size=preferred_size
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
        pytorch_transforms += [torchvision.transforms.ToTensor()]
        
        self.default_train_transform = torchvision.transforms.Compose(pytorch_transforms)
        self.default_test_transform = torchvision.transforms.Compose(pytorch_transforms)
        
        self.train = torchvision.datasets.FashionMNIST(dataset_folder, train=True, download=True, transform=self.default_train_transform)
        self.test = torchvision.datasets.FashionMNIST(dataset_folder, train=False, download=True, transform=self.default_test_transform)
        
        if self.preferred_size == 0:
            self.preferred_size = self.train[0][0].shape[1:]
        
        return
    
    
    def get_class_names(self):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
