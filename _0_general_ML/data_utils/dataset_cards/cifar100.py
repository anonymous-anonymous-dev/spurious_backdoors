import numpy as np
import torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.local_config import dataset_folder



class CIFAR100(Torch_Dataset):
    
    def __init__(
        self,
        preferred_size: int=32,
        data_means = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
        data_stds = [0.24703225141799082, 0.24348516474564, 0.26158783926049628],
        **kwargs
    ):
        
        super().__init__(
            data_name='cifar100',
            preferred_size=preferred_size,
            data_means=data_means,
            data_stds=data_stds
        )
        
        self.renew_data()
        self.num_classes = 100
        
        return
    
    
    def renew_data(
        self, **kwargs
    ):
        
        test_transform = []
        if self.preferred_size:
            test_transform = [torchvision.transforms.Resize(self.preferred_size)]
        test_transform += [torchvision.transforms.ToTensor()] # convert the image to tensor so that it can work with torch
        test_transform +=  [torchvision.transforms.Normalize(tuple(self.data_means), tuple(self.data_stds))] #Normalize all the images
        
        train_transform = []
        if self.preferred_size:
            train_transform = [torchvision.transforms.Resize(self.preferred_size)]
        train_transform += [torchvision.transforms.RandomHorizontalFlip()] # FLips the image w.r.t horizontal axis
        train_transform += [torchvision.transforms.RandomRotation((-7,7))]     #Rotates the image to a specified angel
        train_transform += [torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8,1.2))] #Performs actions like zooms, change shear angles.
        train_transform += [torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)] # Set the color params
        train_transform += [torchvision.transforms.ToTensor()] # convert the image to tensor so that it can work with torch
        train_transform +=  [torchvision.transforms.Normalize(tuple(self.data_means), tuple(self.data_stds))] #Normalize all the images
        # train_transform += [torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        
        self.default_train_transform = torchvision.transforms.Compose(train_transform)
        self.default_test_transform = torchvision.transforms.Compose(test_transform)
        
        self.train = torchvision.datasets.CIFAR100(dataset_folder, train=True, download=True, transform=self.default_train_transform)
        self.test = torchvision.datasets.CIFAR100(dataset_folder, train=False, download=True, transform=self.default_test_transform)
        
        if self.preferred_size == 0:
            self.preferred_size = self.train[0][0].shape[1:]
        
        return


    def get_class_names(self):
        # return np.arange(len(np.unique( [self.train[i][1] for i in range(self.train.__len__())] ))).tolist()
        # return np.arange(100).tolist()
        return [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm"
        ]
