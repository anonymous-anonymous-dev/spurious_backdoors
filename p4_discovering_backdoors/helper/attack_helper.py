from _0_general_ML.model_utils.torch_model import Torch_Model

from ..adversarial_attacks_for_detection.all_available_attacks import Adversarial_Attack_for_Discovering_Backdoors
from ..adversarial_attacks_for_detection.all_available_attacks import Universal_Attack, Trigger_Attack



def get_attack(
    global_model: Torch_Model,
    my_attack_configuration: dict,
    **kwargs
) -> Adversarial_Attack_for_Discovering_Backdoors:
    
    attack_configuration_ = {
        'type': 'trigger_inversion_attack',
        'alpha': 1.
    }
    for key in my_attack_configuration.keys():
        attack_configuration_[key] = my_attack_configuration[key]
    
    all_available_attacks = {
        'universal_adversarial_perturbation': Universal_Attack,
        'trigger_inversion_attack': Trigger_Attack
    }
    
    return all_available_attacks[attack_configuration_['type']](
        global_model,
        attack_configuration=attack_configuration_
    )

