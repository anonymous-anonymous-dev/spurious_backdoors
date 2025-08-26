target_class = 0


general_configuration = {
    'target_class': target_class,
    'num_target_class_samples': 10,
}


snpca_configuration={
    'target_class': target_class, 'non_target_class': None,
    'num_target_class_samples': 10,
    'wrap_and_normalize_model': True,
    'pca_type': 'pca_sklearn',
    'n_components': None,
    'pca_standardization': False,
    'subset_population': 20, 
    'repititions': 100, 
    'sub_pca_components': 1,
    'number_of_best_and_worst_features': 3, 
    'number_of_maximizing_samples_for_evaluating_feature_score': 10,
    'processing_type': '',
    
    # custom with masking configuration
    'masking_configuration': {
        'alpha': 0.3,
        'mask_ratio': 0.7,
        'patch_size': 0.2,
        'iterations': 100,
    }
}


snpca_random_configuration={
    
    # configuration used by NPCA paper
    'target_class': target_class, 'non_target_class': 'random',
    'num_target_class_samples': 10,
    'wrap_and_normalize_model': True,
    'pca_type': 'pca_sklearn',
    'n_components': None,
    'pca_standardization': False,
    'subset_population': None, 
    'repititions': 100, 
    'sub_pca_components': 1,
    'number_of_best_and_worst_features': 3, 
    'number_of_maximizing_samples_for_evaluating_feature_score': 10,
    'processing_type': '', 
    
    # id data samples access
    'num_id_samples': 10,
    
    # custom with masking configuration
    'masking_configuration': {
        'alpha': 0.3,
        'mask_ratio': 0.7,
        'patch_size': 0.2,
        'iterations': 100,
    },
    
    # here are the random configurations
    'fgsm_configuration': {
        'loss': 'crossentropy',
        'epsilon': 0.2,
        'iterations': 50,
    },
    'patch_configuration': {
        'loss': 'crossentropy',
        'mask_ratio': 0.4,
        # 'number_of_masks': 3,
        'iterations': 100,
    },
    
}


defense_configurations = {
    'vanilla': general_configuration,
    'strip': general_configuration,
    'activation_clustering': general_configuration,
    'spectral_signatures': general_configuration,
    'neural_cleanse': general_configuration,
    'mdtd': general_configuration,
    'zero_shot_purification': general_configuration,
    
    'snpca_id': snpca_configuration,
    'snpca_ood': snpca_random_configuration
}


defenses_configured = {
    'vanilla': {'type': 'vanilla'},
    'strip': {'type': 'strip'},
    'activation_clustering': {'type': 'activation_clustering'},
    'spectral_signatures': {'type': 'spectral_signatures'},
    'neural_cleanse': {'type': 'neural_cleanse'},
    'mdtd': {'type': 'mdtd'},
    'zero_shot_purification': {'type': 'zero_shot_purification'},
    
    'snpca_id': {'type': 'snpca_id'},
    'snpca_ood': {'type': 'snpca_ood'},
    
    # =========================
    # Hyperparameters for State of the Art Defenses
    # =========================
    
    # STRIP
    'strip_10': {'type': 'strip', 'num_target_class_samples': 10},
    'strip_20': {'type': 'strip', 'num_target_class_samples': 20},
    'strip_30': {'type': 'strip', 'num_target_class_samples': 30},
    'strip_40': {'type': 'strip', 'num_target_class_samples': 40},
    'strip_50': {'type': 'strip', 'num_target_class_samples': 50},
    'strip_100': {'type': 'strip', 'num_target_class_samples': 100},
    
    # MDTD
    'mdtd_10': {'type': 'mdtd', 'num_target_class_samples': 10},
    'mdtd_20': {'type': 'mdtd', 'num_target_class_samples': 20},
    'mdtd_30': {'type': 'mdtd', 'num_target_class_samples': 30},
    'mdtd_40': {'type': 'mdtd', 'num_target_class_samples': 40},
    'mdtd_50': {'type': 'mdtd', 'num_target_class_samples': 50},
    'mdtd_100': {'type': 'mdtd', 'num_target_class_samples': 100},
    
    # AC
    'activation_clustering_10': {'type': 'activation_clustering', 'num_target_class_samples': 10},
    'activation_clustering_20': {'type': 'activation_clustering', 'num_target_class_samples': 20},
    'activation_clustering_30': {'type': 'activation_clustering', 'num_target_class_samples': 30},
    'activation_clustering_40': {'type': 'activation_clustering', 'num_target_class_samples': 40},
    'activation_clustering_50': {'type': 'activation_clustering', 'num_target_class_samples': 50},
    'activation_clustering_100': {'type': 'activation_clustering', 'num_target_class_samples': 100},
    
    
    # =========================
    # ASNPCA Configurations for Hyperparameter Evaluation
    # =========================
    
    # ---------- ID ----------------
    
    # Hyperparameter - ID - repititions
    'snpca_id_(subset_population=None)_(repititions=1)': {'type': 'snpca_id', 'subset_population': None, 'repititions': 1},
    'snpca_id_(repititions=10)': {'type': 'snpca_id', 'repititions': 10},
    'snpca_id_(repititions=50)': {'type': 'snpca_id', 'repititions': 50},
    'snpca_id_(repititions=100)': {'type': 'snpca_id', 'repititions': 100},
    'snpca_id_(repititions=500)': {'type': 'snpca_id', 'repititions': 500},
    'snpca_id_(repititions=1000)': {'type': 'snpca_id', 'repititions': 1000},
    
    # Hyperparameter - ID - subset population
    'snpca_id_(subset_population=5)': {'type': 'snpca_id', 'subset_population': 5},
    'snpca_id_(subset_population=10)': {'type': 'snpca_id', 'subset_population': 10},
    'snpca_id_(subset_population=20)': {'type': 'snpca_id', 'subset_population': 20},
    'snpca_id_(subset_population=50)': {'type': 'snpca_id', 'subset_population': 50},
    'snpca_id_(subset_population=100)': {'type': 'snpca_id', 'subset_population': 100},
    
    # Hyperparameter - ID - Adversarial Epsilon
    'snpca_id_(adversarial_epsilon=0)': {'type': 'snpca_id', 'adversarial_epsilon': 0.},
    'snpca_id_(adversarial_epsilon=0.1)': {'type': 'snpca_id', 'adversarial_epsilon': 0.1},
    'snpca_id_(adversarial_epsilon=0.2)': {'type': 'snpca_id', 'adversarial_epsilon': 0.2},
    'snpca_id_(adversarial_epsilon=0.3)': {'type': 'snpca_id', 'adversarial_epsilon': 0.3},
    'snpca_id_(adversarial_epsilon=0.4)': {'type': 'snpca_id', 'adversarial_epsilon': 0.4},
    'snpca_id_(adversarial_epsilon=0.5)': {'type': 'snpca_id', 'adversarial_epsilon': 0.5},
    
    # Hyperparameter - ID - num target class samples
    'snpca_id_(accessible_samples=5)': {'type': 'snpca_id', 'num_target_class_samples': 5},
    'snpca_id_(accessible_samples=10)': {'type': 'snpca_id', 'num_target_class_samples': 10},
    'snpca_id_(accessible_samples=20)': {'type': 'snpca_id', 'num_target_class_samples': 20},
    'snpca_id_(accessible_samples=30)': {'type': 'snpca_id', 'num_target_class_samples': 30},
    'snpca_id_(accessible_samples=40)': {'type': 'snpca_id', 'num_target_class_samples': 40},
    'snpca_id_(accessible_samples=50)': {'type': 'snpca_id', 'num_target_class_samples': 50},
    
    
    # ---------- OOD ----------------
    
    # Hyperparameter - OOD - repititions
    'snpca_ood_(subset_population=None)_(repititions=1)': {'type': 'snpca_ood', 'subset_population': None, 'repititions': 1},
    'snpca_ood_(repititions=10)': {'type': 'snpca_ood', 'subset_population': 20, 'repititions': 10},
    'snpca_ood_(repititions=50)': {'type': 'snpca_ood', 'subset_population': 20, 'repititions': 50},
    'snpca_ood_(repititions=100)': {'type': 'snpca_ood', 'subset_population': 20, 'repititions': 100},
    'snpca_ood_(repititions=500)': {'type': 'snpca_ood', 'subset_population': 20, 'repititions': 500},
    'snpca_ood_(repititions=1000)': {'type': 'snpca_ood', 'subset_population': 20, 'repititions': 1000},
    
    # Hyperparameter - OOD - subset population
    'snpca_ood_(subset_population=5)': {'type': 'snpca_ood', 'subset_population': 5},
    'snpca_ood_(subset_population=10)': {'type': 'snpca_ood', 'subset_population': 10},
    'snpca_ood_(subset_population=20)': {'type': 'snpca_ood', 'subset_population': 20},
    'snpca_ood_(subset_population=50)': {'type': 'snpca_ood', 'subset_population': 50},
    'snpca_ood_(subset_population=100)': {'type': 'snpca_ood', 'subset_population': 100},
    
    # Hyperparameter - OOD - Adversarial Epsilon
    'snpca_ood_(adversarial_epsilon=0)': {'type': 'snpca_ood', 'adversarial_epsilon': 0.},
    'snpca_ood_(adversarial_epsilon=0.1)': {'type': 'snpca_ood', 'adversarial_epsilon': 0.1},
    'snpca_ood_(adversarial_epsilon=0.2)': {'type': 'snpca_ood', 'adversarial_epsilon': 0.2},
    'snpca_ood_(adversarial_epsilon=0.3)': {'type': 'snpca_ood', 'adversarial_epsilon': 0.3},
    'snpca_ood_(adversarial_epsilon=0.4)': {'type': 'snpca_ood', 'adversarial_epsilon': 0.4},
    'snpca_ood_(adversarial_epsilon=0.5)': {'type': 'snpca_ood', 'adversarial_epsilon': 0.5},
    
    # Hyperparameter - OOD - masking ratio
    'snpca_ood_(mask_ratio=0.3)': {'type': 'snpca_ood', 'masking_configuration': {'mask_ratio': 0.3},},
    'snpca_ood_(mask_ratio=0.4)': {'type': 'snpca_ood', 'masking_configuration': {'mask_ratio': 0.4},},
    'snpca_ood_(mask_ratio=0.5)': {'type': 'snpca_ood', 'masking_configuration': {'mask_ratio': 0.5},},
    'snpca_ood_(mask_ratio=0.6)': {'type': 'snpca_ood', 'masking_configuration': {'mask_ratio': 0.6},},
    'snpca_ood_(mask_ratio=0.7)': {'type': 'snpca_ood', 'masking_configuration': {'mask_ratio': 0.7},},
    
    # Hyperparameter - OOD - patch_ratio when creating targeted adversarial patch at random locations
    'snpca_ood_(patch_ratio=0.3)': {'type': 'snpca_ood', 'patch_configuration': {'mask_ratio': 0.3},},
    'snpca_ood_(patch_ratio=0.4)': {'type': 'snpca_ood', 'patch_configuration': {'mask_ratio': 0.4},},
    'snpca_ood_(patch_ratio=0.5)': {'type': 'snpca_ood', 'patch_configuration': {'mask_ratio': 0.5},},
    'snpca_ood_(patch_ratio=0.6)': {'type': 'snpca_ood', 'patch_configuration': {'mask_ratio': 0.6},},
    'snpca_ood_(patch_ratio=0.7)': {'type': 'snpca_ood', 'patch_configuration': {'mask_ratio': 0.7},},
    
    # Hyperparameter - OOD - Number of clean samples
    'snpca_ood_10': {'type': 'snpca_ood', 'num_target_class_samples': 10},
    'snpca_ood_20': {'type': 'snpca_ood', 'num_target_class_samples': 20},
    'snpca_ood_30': {'type': 'snpca_ood', 'num_target_class_samples': 30},
    'snpca_ood_40': {'type': 'snpca_ood', 'num_target_class_samples': 40},
    'snpca_ood_50': {'type': 'snpca_ood', 'num_target_class_samples': 50},
    'snpca_ood_100': {'type': 'snpca_ood', 'num_target_class_samples': 100},
    
    # Hyperparameter - OOD - num target class samples
    'snpca_ood_(accessible_samples=5)': {'type': 'snpca_ood', 'num_target_class_samples': 5},
    'snpca_ood_(accessible_samples=10)': {'type': 'snpca_ood', 'num_target_class_samples': 10},
    'snpca_ood_(accessible_samples=20)': {'type': 'snpca_ood', 'num_target_class_samples': 20},
    'snpca_ood_(accessible_samples=30)': {'type': 'snpca_ood', 'num_target_class_samples': 30},
    'snpca_ood_(accessible_samples=40)': {'type': 'snpca_ood', 'num_target_class_samples': 40},
    'snpca_ood_(accessible_samples=50)': {'type': 'snpca_ood', 'num_target_class_samples': 50},
    
    # OOD play
    'snpca_ood_efficient': {'type': 'snpca_ood', 'fgsm_configuration': {'iterations': 20}, 'masking_configuration': {'iterations': 50}, 'patch_configuration': {'iterations': 50},},
    
}

