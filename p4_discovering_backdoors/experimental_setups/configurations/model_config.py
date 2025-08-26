mnist_toy_model_configuration = {
    'model_architecture': 'mnist_cnn',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 50,
    'batch_size': 512,
    'optimizer': 'adam',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 100,
}
mnist_model_configuration = {
    'model_architecture': 'mnist_cnn',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 50,
    'batch_size': 1024,
    'optimizer': 'adam',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
}


cifar10_model_configuration = {
    'model_architecture': 'cifar10_resnet18',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': [200, 70],
    'batch_size': 512,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 150,
}
cifar10_vit_model_configuration = {
    'model_architecture': 'cifar10_vit16',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 70,
    'batch_size': 512,
    'optimizer': 'adam',
    'momentum': 0.9,
    'weight_decay': 0, # 5e-4,
    'patience': 150,
}
cifar10_model_configuration_non_sota = {
    'model_architecture': 'cifar10_resnet18',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 50,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 70,
}


cifar100_model_configuration = {
    'model_architecture': 'cifar100_resnet18',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': [500, 70],
    'batch_size': 512,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 150,
}
cifar100_vit_model_configuration = {
    'model_architecture': 'cifar100_vit16',
    'learning_rate': 1e-4,
    'loss_fn': 'crossentropy',
    'epochs': 70,
    'batch_size': 512,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 150,
}
cifar100_model_configuration_non_sota = {
    'model_architecture': 'cifar100_resnet18',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 100,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 70,
}


gtsrb_model_configuration = {
    'model_architecture': 'resnet18_gtsrb',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': [100, 50],
    'batch_size': 256,
    'optimizer': 'adam',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 100,
}
gtsrb_model_configuration_non_sota = {
    'model_architecture': 'resnet18_gtsrb',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 50,
    'batch_size': 256,
    'optimizer': 'adam',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
}


imagenet_model_configuration_resnet18 = {
    'model_architecture': 'kaggle_imagenet_resnet18',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 100,
    'batch_size': 128,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 100,
}
imagenet_model_configuration_resnet50 = {
    'model_architecture': 'kaggle_imagenet_resnet50',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 100,
    'batch_size': 128,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 100,
}
imagenet_model_configuration_vit_b_16 = {
    'model_architecture': 'kaggle_imagenet_vit_b_16',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 100,
    'batch_size': 64,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 100,
}


# some configurations for TM2 and TM3
mnist_fashion_model_configuration = {
    'model_architecture': 'mnist_cnn',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 50,
    'batch_size': 1024,
    'optimizer': 'adam',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
    'old_dataset_name': 'mnist',
    'new_dataset_name': 'fashion_mnist',
    'train_size': 5000,
}
cifar10_gtsrb_model_configuration = {
    'model_architecture': 'cifar10_resnet18',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 200,
    'batch_size': 512,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 150,
    'old_dataset_name': 'cifar10',
    'new_dataset_name': 'gtsrb',
    'train_size': 5000,
}
cifar10_mnist_model_configuration = {
    'model_architecture': 'cifar10_resnet18',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 200,
    'batch_size': 512,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 150,
    'old_dataset_name': 'cifar10',
    'new_dataset_name': 'mnist3',
    'train_size': 2000,
}
cifar100_cifar10_model_configuration = {
    'model_architecture': 'cifar100_resnet18',
    'learning_rate': 1e-2,
    'loss_fn': 'crossentropy',
    'epochs': 500,
    'batch_size': 512,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 150,
    'old_dataset_name': 'cifar100',
    'new_dataset_name': 'cifar10',
    'train_size': 5000,
}


model_configurations = {
    'mnist_toy': mnist_toy_model_configuration,
    'mnist': mnist_model_configuration,
    'gtsrb': gtsrb_model_configuration,
    'cifar10': cifar10_model_configuration,
    'cifar100': cifar100_model_configuration,
    'kaggle_imagenet_R50': imagenet_model_configuration_resnet50,
    'kaggle_imagenet_R18': imagenet_model_configuration_resnet18,
    'kaggle_imagenet_vit_b_16': imagenet_model_configuration_vit_b_16,
    # advanced_models
    'cifar10_vit16': cifar10_vit_model_configuration,
    'cifar100_vit16': cifar100_vit_model_configuration,
    'cifar10_vit16_official': cifar10_vit_model_configuration,
    'cifar100_vit16_official': cifar100_vit_model_configuration,
    # model reuse settings
    'mnist_fashion': mnist_fashion_model_configuration,
    'cifar10_gtsrb': cifar10_gtsrb_model_configuration,
    'cifar100_cifar10': cifar100_cifar10_model_configuration,
    # model finetune settings
    'cifar10_mnist': cifar10_mnist_model_configuration,
    'cifar100_cifar10': cifar100_cifar10_model_configuration,
    # non standard settings
    'cifar10_non_sota': cifar10_model_configuration_non_sota,
    'gtsrb_non_sota': gtsrb_model_configuration_non_sota,
}
