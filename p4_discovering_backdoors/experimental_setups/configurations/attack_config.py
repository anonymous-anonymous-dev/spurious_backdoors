# ##############################
# SOME COMMENTS REGARDING THE CONFIGURATION VARIABLES
# ##############################
# 
# ##############################
# <target>
# ##############################
# <target> can be <None>, <all> or <any integer class>
# <target> = <None> carries out an untargeted backdoor detection
# <target> = <an integer number> detects/highlights backdoor in the target class
# <target> = <all> detects/highlights backdoors in all possible target classes.
# ##############################
# 


universal_attack_default_configuration = {
    'iterations': 5000,
    'epsilon': 0.5,
    'target': None
}

trigger_attack_default_configuration = {
    'iterations': 5000,
    'epsilon': 1.,
    'alpha': 1.,
    'target': None
}


# attack configuration dictionary
attack_configurations = {
    'universal_adversarial_perturbation': universal_attack_default_configuration,
    'trigger_inversion_attack': trigger_attack_default_configuration
}


# different attacks configured
different_attacks_configured = {
    
    'uap_default': {'type': 'universal_adversarial_perturbation'},
    'uap_targeted': {'type': 'universal_adversarial_perturbation', 'target': 'all'},
    
    'tia_default': {'type': 'trigger_inversion_attack'},
    'tia_targeted': {'type': 'trigger_inversion_attack', 'target': 'all'},
    'tia_targeted_alphas': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 'configurable'},
    
    'tia_(target=0)_alphas': {'type': 'trigger_inversion_attack', 'target': 0, 'alpha': 'configurable'},
    'tia_(target=1)_alphas': {'type': 'trigger_inversion_attack', 'target': 1, 'alpha': 'configurable'},
    'tia_(target=2)_alphas': {'type': 'trigger_inversion_attack', 'target': 2, 'alpha': 'configurable'},
    'tia_(target=3)_alphas': {'type': 'trigger_inversion_attack', 'target': 3, 'alpha': 'configurable'},
    'tia_(target=4)_alphas': {'type': 'trigger_inversion_attack', 'target': 4, 'alpha': 'configurable'},
    'tia_(target=5)_alphas': {'type': 'trigger_inversion_attack', 'target': 5, 'alpha': 'configurable'},
    'tia_(target=6)_alphas': {'type': 'trigger_inversion_attack', 'target': 6, 'alpha': 'configurable'},
    'tia_(target=7)_alphas': {'type': 'trigger_inversion_attack', 'target': 7, 'alpha': 'configurable'},
    'tia_(target=8)_alphas': {'type': 'trigger_inversion_attack', 'target': 8, 'alpha': 'configurable'},
    'tia_(target=9)_alphas': {'type': 'trigger_inversion_attack', 'target': 9, 'alpha': 'configurable'},
    
    'tia_(target=0)_(alpha=0.03)': {'type': 'trigger_inversion_attack', 'target': 0, 'alpha': 0.03},
    'tia_(target=1)_(alpha=0.03)': {'type': 'trigger_inversion_attack', 'target': 1, 'alpha': 0.03},
    'tia_(target=2)_(alpha=0.03)': {'type': 'trigger_inversion_attack', 'target': 2, 'alpha': 0.03},
    'tia_(target=3)_(alpha=0.03)': {'type': 'trigger_inversion_attack', 'target': 3, 'alpha': 0.03},
    'tia_(target=4)_(alpha=0.03)': {'type': 'trigger_inversion_attack', 'target': 4, 'alpha': 0.03},
    'tia_(target=5)_(alpha=0.03)': {'type': 'trigger_inversion_attack', 'target': 5, 'alpha': 0.03},
    'tia_(target=6)_(alpha=0.03)': {'type': 'trigger_inversion_attack', 'target': 6, 'alpha': 0.03},
    'tia_(target=7)_(alpha=0.03)': {'type': 'trigger_inversion_attack', 'target': 7, 'alpha': 0.03},
    'tia_(target=8)_(alpha=0.03)': {'type': 'trigger_inversion_attack', 'target': 8, 'alpha': 0.03},
    'tia_(target=9)_(alpha=0.03)': {'type': 'trigger_inversion_attack', 'target': 9, 'alpha': 0.03},
    
    'tia_(target=0)_(alpha=0.01)': {'type': 'trigger_inversion_attack', 'target': 0, 'alpha': 0.01},
    'tia_(target=1)_(alpha=0.01)': {'type': 'trigger_inversion_attack', 'target': 1, 'alpha': 0.01},
    'tia_(target=2)_(alpha=0.01)': {'type': 'trigger_inversion_attack', 'target': 2, 'alpha': 0.01},
    'tia_(target=3)_(alpha=0.01)': {'type': 'trigger_inversion_attack', 'target': 3, 'alpha': 0.01},
    'tia_(target=4)_(alpha=0.01)': {'type': 'trigger_inversion_attack', 'target': 4, 'alpha': 0.01},
    'tia_(target=5)_(alpha=0.01)': {'type': 'trigger_inversion_attack', 'target': 5, 'alpha': 0.01},
    'tia_(target=6)_(alpha=0.01)': {'type': 'trigger_inversion_attack', 'target': 6, 'alpha': 0.01},
    'tia_(target=7)_(alpha=0.01)': {'type': 'trigger_inversion_attack', 'target': 7, 'alpha': 0.01},
    'tia_(target=8)_(alpha=0.01)': {'type': 'trigger_inversion_attack', 'target': 8, 'alpha': 0.01},
    'tia_(target=9)_(alpha=0.01)': {'type': 'trigger_inversion_attack', 'target': 9, 'alpha': 0.01},
    
    'tia_targeted_(alpha=0.010)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.010},
    'tia_targeted_(alpha=0.012)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.012},
    'tia_targeted_(alpha=0.014)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.014},
    'tia_targeted_(alpha=0.016)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.016},
    'tia_targeted_(alpha=0.018)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.018},
    'tia_targeted_(alpha=0.020)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.020},
    'tia_targeted_(alpha=0.022)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.022},
    'tia_targeted_(alpha=0.024)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.024},
    'tia_targeted_(alpha=0.026)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.026},
    'tia_targeted_(alpha=0.028)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.028},
    'tia_targeted_(alpha=0.030)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.030},
    'tia_targeted_(alpha=0.032)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.032},
    'tia_targeted_(alpha=0.034)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.034},
    'tia_targeted_(alpha=0.036)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.036},
    'tia_targeted_(alpha=0.038)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.038},
    'tia_targeted_(alpha=0.040)': {'type': 'trigger_inversion_attack', 'target': 'all', 'alpha': 0.040},
    
}