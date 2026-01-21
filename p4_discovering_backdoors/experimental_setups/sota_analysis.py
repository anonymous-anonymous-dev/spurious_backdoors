# Dataset to perform the analysis on
dataset_names = [
    # 'mnist',
    'cifar10',
    'cifar10_convnext',
    'cifar100',
    'cifar100_convnext',
    
    # 'cifar10_vit16',
    # 'cifar100_vit16',
    
    # 'gtsrb',
    
    # 'kaggle_imagenet_R18',
    # 'kaggle_imagenet_R50',
    # 'kaggle_imagenet_VT_B16'
    
    # 'cifar10_mnist',
    # 'cifar100_cifar10'
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
    'wanet_backdoor_1.0',
    
    # 'invisible_backdoor_1.0',
    'clean_label_backdoor_1.0',
    # 'horizontal_backdoor_1.0',
    
    # 'invisible_backdoor_3.0',
    'clean_label_backdoor_3.0',
    # 'horizontal_backdoor_3.0',
    
    # 'invisible_backdoor_5.0',
    'clean_label_backdoor_5.0',
    # 'horizontal_backdoor_5.0',
    
    # 'wanet_backdoor_2.0',
    # 'wanet_backdoor_5.0',
    # 'wanet_backdoor_10.0',
    # 'reflection_backdoor_1.0',
    # 'reflection_backdoor_3.0',
    # 'reflection_backdoor_5.0',
    
    # 'reflection_backdoor_0.5',
    # 'reflection_backdoor_0.6',
    # 'reflection_backdoor_0.7',
    # 'reflection_backdoor_0.8',
    # 'reflection_backdoor_0.9',
    # 'reflection_backdoor_1.0',
    
    # 'wanet_backdoor_0.5',
    # 'wanet_backdoor_0.6',
    # 'wanet_backdoor_0.7',
    # 'wanet_backdoor_0.8',
    # 'wanet_backdoor_0.9',
    # 'wanet_backdoor_1.0',
    # 'wanet_backdoor_1.3',
    
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


