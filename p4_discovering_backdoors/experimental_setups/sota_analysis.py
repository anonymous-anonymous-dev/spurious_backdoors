# Dataset to perform the analysis on
mf_dataset_names = [
    'cifar10_mnist',
    'cifar100_cifar10',
]

# mr_dataset_names = [
#     # 'mnist_fashion',
#     # 'cifar10_gtsrb',
#     # 'cifar100_cifar10',
# ]

dataset_names = [
    'mnist',
    'cifar10',
    'cifar100',
    # 'gtsrb',
    
    # 'cifar10_vit16',
    # 'cifar100_vit16',
    
    # 'kaggle_imagenet_R18',
    # 'kaggle_imagenet_R50',
    # 'kaggle_imagenet_VT_B16'
]


backdoor_attack_types = [
    
    # 'spurious_0.1',
    # 'spurious_0.3',
    # 'spurious_sun_0.1',
    
    'simple_backdoor_0',
    
    # 'simple_backdoor_0.1',
    # 'invisible_backdoor_0.1',
    # 'reflection_backdoor_0.1',
    # 'clean_label_backdoor_0.1',
    # 'wanet_backdoor_0.1',
    # 'horizontal_backdoor_0.1',
    
    'simple_backdoor_0.3',
    'invisible_backdoor_0.3',
    'reflection_backdoor_0.3',
    'wanet_backdoor_0.3',
    'clean_label_backdoor_0.3',
    'horizontal_backdoor_0.3',
    
]


defense_types = [
    'vanilla',
    'strip', 
    'activation_clustering', 
    'spectral_signatures',
    'mdtd',
    'zero_shot_purification',
    
    'snpca_id',
    'snpca_ood'
]

