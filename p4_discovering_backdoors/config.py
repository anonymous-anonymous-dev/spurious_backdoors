from _0_general_ML.configurations import model_config_1 as model_configurations_config

# from .experimental_setups.configurations import model_config as model_configurations_config
from .experimental_setups.configurations import attack_config as attack_configurations_config
from .experimental_setups.configurations import backdoor_attack_config as backdoor_configurations_config
from .experimental_setups.configurations import defense_config as defense_configurations_config

from .experimental_setups import hyperparameter_pr, hyperparameter_asnpca, sota_analysis, global_aug_sota_analysis



# experimental setup
experimental_setups = [
    sota_analysis,
    # global_aug_sota_analysis,
    # hyperparameter_pr,
    # hyperparameter_asnpca,
]

#########################
# Visible GPU
visible_gpu = '1'
multiprocessing_shot = False
shots_at_a_time = 1
wait_before_starting_the_next_process = 0 # in seconds
versioning = False

# General configurations
experiment_folder = 'results_snpca_3/'
results_path = '../../__all_results__/_p4_discovering_backdoors/' + experiment_folder
reconduct_conducted_experiments = False
count_continued_as_conducted = False
save_continued = False
force_overwrite_csv_results = False

# Data configurations
dataset_folder = '../../_Datasets/'

# Backdooring configurations
all_backdoor_configurations = backdoor_configurations_config.backdoor_configurations
configured_backdoors = backdoor_configurations_config.backdoor_attacks_configured

# Model configurations
model_configurations = model_configurations_config.model_configurations
test_model_before_backdoor_evaluation = False

# Attack configurations
attack_configurations = attack_configurations_config.attack_configurations
configured_attacks = attack_configurations_config.different_attacks_configured

# Defense configurations
all_defense_configurations = defense_configurations_config.defense_configurations
configured_defenses = defense_configurations_config.defenses_configured

# # Evaluation configurations
# evaluation_configurations = imagenet_config.evaluation_configs

