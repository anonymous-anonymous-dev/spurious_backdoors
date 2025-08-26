import torch
import torchvision
import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.local_config import dataset_folder



class CelebA(Torch_Dataset):
    
    def __init__(
        self,
        preferred_size: int=(64, 64),
        data_means: list[int]=[0.5, 0.5, 0.5],
        data_stds: list[int]=[0.5, 0.5, 0.5],
        **kwargs
    ):
        
        super().__init__(
            data_name='gtsrb',
            preferred_size=preferred_size,
            data_means=data_means,
            data_stds=data_stds
        )
        
        if not self.preferred_size:
            self.preferred_size = (64, 64)
        
        self.renew_data()
        # self.num_classes = len(self.get_class_names())
        self.num_classes = 40
        
        return
    
    
    def renew_data(
        self, **kwargs
    ):
        
        pytorch_transforms = []
        if self.preferred_size:
            pytorch_transforms = [torchvision.transforms.Resize(self.preferred_size)]
        pytorch_transforms += [torchvision.transforms.ToTensor()]
        pytorch_transforms += [torchvision.transforms.Normalize(tuple(self.data_means), tuple(self.data_stds))]
        
        self.default_train_transform = torchvision.transforms.Compose(pytorch_transforms)
        self.default_test_transform = torchvision.transforms.Compose(pytorch_transforms)
        
        self.train = torchvision.datasets.CelebA(dataset_folder, split='train', download=True, transform=self.default_train_transform)
        self.test = torchvision.datasets.CelebA(dataset_folder, split='test', download=True, transform=self.default_test_transform)
        
        self.train.targets = [self.train[i][1] for i in range(self.train.__len__())]
        self.test.targets = [self.test[i][1] for i in range(self.test.__len__())]
        
        if self.preferred_size == 0:
            self.preferred_size = self.train[0][0].shape[1:]
        
        return
    
    
    def get_class_names(self):
        return list(np.arange(len(np.unique( [self.train[i][1] for i in range(self.train.__len__())] ))))
    
    