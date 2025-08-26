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
    # 'mnist',
    # 'cifar10',
    # 'cifar100',
    'gtsrb',
    
    'cifar10_vit16',
    'cifar100_vit16',
    
    # 'kaggle_imagenet_R18',
    # 'kaggle_imagenet_R50',
    # 'kaggle_imagenet_VT_B16'
]


backdoor_attack_types = [
    
    'simple_backdoor_0',
    
    # 'simple_backdoor_0.1',
    # 'invisible_backdoor_0.1',
    # 'reflection_backdoor_0.1',
    # # 'clean_label_backdoor_0.1',
    # 'wanet_backdoor_0.1',
    # 'horizontal_backdoor_0.1',
    
    'simple_backdoor_0.3',
    'invisible_backdoor_0.3',
    'reflection_backdoor_0.3',
    'wanet_backdoor_0.3',
    # 'clean_label_backdoor_0.3',
    'horizontal_backdoor_0.3',
    
    # global
    # 'simple_backdoor_0.01_global_poisoning',
    # 'invisible_backdoor_0.01_global_poisoning',
    # 'reflection_backdoor_0.01_global_poisoning',
    # # 'clean_label_backdoor_0.01_global_poisoning',
    # 'wanet_backdoor_0.01_global_poisoning',
    # 'horizontal_backdoor_0.01_global_poisoning',
    
    'simple_backdoor_0.03_global_poisoning',
    'invisible_backdoor_0.03_global_poisoning',
    'reflection_backdoor_0.03_global_poisoning',
    # 'clean_label_backdoor_0.03_global_poisoning',
    'wanet_backdoor_0.03_global_poisoning',
    'horizontal_backdoor_0.03_global_poisoning',
    
]


defense_types = [
    # 'vanilla',
    # 'strip', 
    # 'activation_clustering', 
    # 'spectral_signatures',
    # 'mdtd',
    # 'zero_shot_purification',
    
    # 'snpca_id',
    'snpca_ood'
]

