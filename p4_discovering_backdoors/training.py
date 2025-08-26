import torch

from _0_general_ML.data_utils.datasets import MNIST, GTSRB, CIFAR10, Kaggle_Imagenet
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor



def main():
    
    my_data = Kaggle_Imagenet()
    print(my_data.num_classes)
    print('Data has been loaded.')
    
    model_architectures = ['resnet18_gtsrb', 'mnist_cnn', 'cifar10_vgg11', 'cifar10_resnet18', 'kaggle_imagenet_resnet50']
    my_model = Torch_Model(
        my_data, 
        model_configuration = {
            'model_architecture': model_architectures[4],
            'learning_rate': 0.1,
            'loss_fn': 'crossentropy',
            'epochs': 1000,
            'batch_size': 256,
            'optimizer': 'sgd',
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'patience': 70,
            'split_type': 'iid'
        }
    )
    
    test_loader = torch.utils.data.DataLoader(my_data.test, shuffle=True, batch_size=my_model.model_configuration['batch_size'])
    my_model.test_shot(test_loader)
    print('\n')
    
    return

