import torch
import numpy as np


from _1_adversarial_ML.backdoor_defenses.post_training_defenses.activation_clustering import Activation_Clustering
from _1_adversarial_ML.backdoor_defenses.post_training_defenses.strip import STRIP
from _1_adversarial_ML.backdoor_defenses.post_training_defenses.spectral_signatures import Spectral_Clustering
from _1_adversarial_ML.backdoor_defenses.post_training_defenses.neural_cleanse import Neural_Cleanse
from _1_adversarial_ML.backdoor_defenses.post_training_defenses.multidomain_trojan_detector import Multi_Domain_Trojan_Detector
from _1_adversarial_ML.backdoor_defenses.post_training_defenses.zero_shot_defense import Zero_Shot_Image_Purification

from _1_adversarial_ML.backdoor_defenses.post_training_defenses.backdoor_defense import Backdoor_Detection_Defense


from ..snpca.npca_custom_no_masking import NPCA_Custom
from ..snpca.npca_custom_with_masking import NPCA_Custom_with_Masking
from ..snpca.npca_random import NPCA_Random
from ..snpca.npca_random_ood import NPCA_Random_OOD
"""
Note that the defense must have a function called evaluate.
This function when called returns the (loss_clean, acc_clean), (loss_poisoned, acc_poisoned) values
"""



defenses_dict = {
    'vanilla': Backdoor_Detection_Defense,
    'strip': STRIP,
    'activation_clustering': Activation_Clustering,
    'spectral_signatures': Spectral_Clustering,
    'neural_cleanse': Neural_Cleanse,
    'mdtd': Multi_Domain_Trojan_Detector,
    'zero_shot_purification': Zero_Shot_Image_Purification,
    
    # our proposed SNPCA
    'snpca_id': NPCA_Custom,
    # 'snpca_random': NPCA_Random,
    'snpca_ood': NPCA_Random
}


def get_defense(
    data, model,
    defense_configuration: dict={},
    **kwargs
) -> Backdoor_Detection_Defense:
    
    defense = defenses_dict[defense_configuration['type']](data=data, torch_model=model, defense_configuration=defense_configuration, **kwargs)
    
    return defense


