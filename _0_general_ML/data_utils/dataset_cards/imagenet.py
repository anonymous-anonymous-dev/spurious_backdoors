import numpy as np
import os
import torch
import torchvision

from termcolor import colored


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset, Client_SubDataset
from _0_general_ML.local_config import dataset_folder

from .imagenet_class_mapping import imagenet_class_mapping_dictionary



class Kaggle_Imagenet(Torch_Dataset):
    
    def __init__(
        self,
        preferred_size: int=(224, 224),
        data_means = [0.485, 0.456, 0.406],
        data_stds = [0.229, 0.224, 0.225],
        **kwargs
    ):
        
        super().__init__(
            data_name='kaggle_imagenet',
            preferred_size=preferred_size,
            data_means=data_means,
            data_stds=data_stds
        )
        
        self.dataset_folder = dataset_folder+'imagenet/ILSVRC/Data/CLS-LOC/'
        
        self.renew_data()
        
        print('Calculating the number of classes...', end='')
        self.num_classes = 1000 # len(self.get_class_names())
        print(f'\rThe number of classes in {self.data_name} is: {self.num_classes}.')
        
        return
    
    
    def renew_data(self, **kwargs):
        
        test_transform = []
        if self.preferred_size:
            print(f'Preferred input size is {self.preferred_size}.')
            test_transform = [torchvision.transforms.Resize(self.preferred_size)]
        test_transform += [torchvision.transforms.ToTensor()] # convert the image to tensor so that it can work with torch
        test_transform += [torchvision.transforms.Normalize(tuple(self.data_means), tuple(self.data_stds))]
        
        train_transform = []
        if self.preferred_size:
            train_transform = [torchvision.transforms.Resize(self.preferred_size)]
        # train_transform += [torchvision.transforms.RandomCrop(32)]
        # train_transform += [torchvision.transforms.RandomHorizontalFlip()]
        train_transform += [torchvision.transforms.ToTensor()]
        train_transform += [torchvision.transforms.Normalize(tuple(self.data_means), tuple(self.data_stds))]
        
        self.default_train_transform = torchvision.transforms.Compose(train_transform)
        self.default_test_transform = torchvision.transforms.Compose(test_transform)
        
        self.full_dataset = torchvision.datasets.ImageFolder(self.dataset_folder+'train/', transform=self.default_train_transform)
        
        # train_size = int(0.9 * len(self.full_dataset))
        # test_size = len(self.full_dataset) - train_size
        # self.train, self.test = torch.utils.data.random_split(self.full_dataset, [train_size, test_size])
        
        filepath_and_name = '__ignore__/imagenet_personal_train_test_split_indices.npz'
        if os.path.isfile(filepath_and_name):
            _train_test_splits = np.load(filepath_and_name)
            train_idcs, test_idcs = _train_test_splits['train_idcs'], _train_test_splits['test_idcs']
        else:
            train_size = int(0.9 * len(self.full_dataset))
            test_size = len(self.full_dataset) - train_size
            all_indices = np.random.choice(len(self.full_dataset), len(self.full_dataset), replace=False)
            train_idcs, test_idcs = all_indices[:train_size], all_indices[train_size:]
            np.savez_compressed(filepath_and_name, train_idcs=train_idcs, test_idcs=test_idcs)
        self.train = Client_SubDataset(self.full_dataset, train_idcs)
        self.test = Client_SubDataset(self.full_dataset, test_idcs)
        # self.train = torch.utils.data.Subset(self.full_dataset, train_idcs)
        # self.test = torch.utils.data.Subset(self.full_dataset, test_idcs)
        # self.train.targets = (torch.tensor(self.full_dataset.targets)[train_idcs]).tolist()
        # self.test.targets = (torch.tensor(self.full_dataset.targets)[test_idcs]).tolist()
        
        if self.preferred_size == 0:
            self.preferred_size = self.train[0][0].shape[1:]
        
        return
    
    
    def update_transforms(self, transform, subdata_category = 'all'):
        if subdata_category in ['all', 'train', 'test']:
            self.full_dataset.transform = transform
            # self.train.data.transform = transform
            # self.test.data.transform = transform
        return
    
    
    def compute_class_names(self):
        
        dataloader = torch.utils.data.DataLoader(self.train, batch_size=256)
        len_dataloader = len(dataloader)
        print('Length of dataloader is: ', len_dataloader)
        
        labels_list = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                labels_list += target.view(-1)
                labels_list = list(np.unique(labels_list))
                
                print_str = f'Processing: [{batch_idx+1}/{len_dataloader}]({(100.*(batch_idx+1)/len_dataloader):3.1f}%). Found classes: {len(labels_list)}'
                print('\r'+print_str, end='')
        
        return np.unique(labels_list)
    
    
    def get_class_names(self): return [f'{imagenet_class_mapping_dictionary[k].split(',')[0]}' for k in imagenet_class_mapping_dictionary.keys()]
    def get_num_classes(self): return self.num_classes
    
    