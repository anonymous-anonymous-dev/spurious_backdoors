# Dataset to perform the analysis on
mf_dataset_names = [
    # 'cifar10_mnist',
    # 'cifar100_cifar10',
]

# mr_dataset_names = [
#     # 'mnist_fashion',
#     # 'cifar10_gtsrb',
#     # 'cifar100_cifar10',
# ]

dataset_names = [
    # 'mnist',
    'cifar10',
    # 'cifar100',
    # 'gtsrb',
    
    # 'kaggle_imagenet_R18',
    # 'kaggle_imagenet_R50',
    # 'kaggle_imagenet_VT_B16'
]


backdoor_attack_types = [
    
    # # 'spurious_0.1',
    # # 'spurious_0.3',
    # # 'spurious_sun_0.1',
    
    # 'simple_backdoor_0',
    
    # 'simple_backdoor_0.1',
    'simple_backdoor_0.3',
    
    # 'invisible_backdoor_0.1',
    # 'invisible_backdoor_0.3',
    
    # 'reflection_backdoor_0.1',
    # 'reflection_backdoor_0.3',
    
    # # 'clean_label_backdoor_0.1',
    # # 'clean_label_backdoor_0.3',
    
    # 'wanet_backdoor_0.1',
    # 'wanet_backdoor_0.3',
    
    # 'horizontal_backdoor_0.1',
    # 'horizontal_backdoor_0.3',
    
]


defense_types = [
    # 'vanilla',
    
    # Hyperparameter - ID - Adversarial Epsilon
    'snpca_id_(adversarial_epsilon=0)',
    'snpca_id_(adversarial_epsilon=0.1)',
    'snpca_id_(adversarial_epsilon=0.2)',
    'snpca_id_(adversarial_epsilon=0.3)',
    'snpca_id_(adversarial_epsilon=0.4)',
    # 'snpca_id_(adversarial_epsilon=0.5)',
    
    
    # -------------- OOD ---------------
    # Hyperparameter - OOD - masking ratio
    'snpca_ood_(mask_ratio=0.3)',
    'snpca_ood_(mask_ratio=0.4)',
    'snpca_ood_(mask_ratio=0.5)',
    'snpca_ood_(mask_ratio=0.6)',
    'snpca_ood_(mask_ratio=0.7)',
    
    # Hyperparameter - OOD - patch_ratio when creating targeted adversarial patch at random locations
    'snpca_ood_(patch_ratio=0.3)',
    'snpca_ood_(patch_ratio=0.4)',
    'snpca_ood_(patch_ratio=0.5)',
    'snpca_ood_(patch_ratio=0.6)',
    'snpca_ood_(patch_ratio=0.7)',
    
    # Hyperparameter - OOD - adversarial epsilon
    'snpca_ood_(adversarial_epsilon=0)',
    'snpca_ood_(adversarial_epsilon=0.1)',
    'snpca_ood_(adversarial_epsilon=0.2)',
    'snpca_ood_(adversarial_epsilon=0.3)',
    'snpca_ood_(adversarial_epsilon=0.4)',
    
    # Hyperparameter - OOD - num target class samples
    'snpca_ood_(accessible_samples=5)',
    'snpca_ood_(accessible_samples=10)',
    'snpca_ood_(accessible_samples=20)',
    'snpca_ood_(accessible_samples=30)',
    'snpca_ood_(accessible_samples=40)',
    'snpca_ood_(accessible_samples=50)',
    
]

