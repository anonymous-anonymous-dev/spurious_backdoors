import numpy as np
import gc

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


from ..config import *

# from .train_shot import training_shot as shot
from .poisoned_shot import poisoned_training_shot as shot

from ..helper.helper_multiprocessing import Helper_Multiprocessing



def sub_main_mp(
    kwargs
):
    
    # *** preparing variables out of kwargs ***
    configuration_variables = kwargs['configuration_variables']
    my_model_configuration = kwargs['my_model_configuration']
    my_backdoor_configuration = kwargs['my_backdoor_configuration']
    
    shot(
        configuration_variables,
        my_model_configuration,
        my_backdoor_configuration
    )
    
    return


def sub_main(
    configuration_variables: dict,
    my_model_configuration: dict,
    my_backdoor_configuration: dict
):
    
    return multiprocessing.Process(
        target = sub_main_mp,
        args = (
            {
                'configuration_variables': configuration_variables,
                'my_model_configuration': my_model_configuration,
                'my_backdoor_configuration': my_backdoor_configuration
            },
        )
    )


def main(threat_models: list[str]=['DC'], orientation=0):
    
    if orientation == 1:
        _experimental_setups = experimental_setups[::-1]
    else:
        _experimental_setups = experimental_setups
        
    # starts here
    current_experiment_number = 0; exceptions_met = 0
    all_processes = []
    for threat_model in threat_models:
        for experimental_setup in _experimental_setups:
            dataset_names = experimental_setup.dataset_names
            backdoor_attack_types = experimental_setup.backdoor_attack_types
            
            total_experiments = len(dataset_names) * len(backdoor_attack_types) * len(threat_models)
            
            # setting the orientation of the experiment
            if orientation == 1:
                _dataset_names = dataset_names[::-1]
            else:
                _dataset_names = dataset_names
            
            # iterating over data
            for dataset_name in _dataset_names:
                
                # iterating over backdoor attacks
                for backdoor_type in backdoor_attack_types:
                        
                    my_model_configuration = model_configurations[dataset_name].copy()
                    my_model_configuration['dataset_name'] = dataset_name
                    
                    my_backdoor_configuration = all_backdoor_configurations[configured_backdoors[backdoor_type]['type']].copy()
                    for key in configured_backdoors[backdoor_type].keys():
                        my_backdoor_configuration[key] = configured_backdoors[backdoor_type][key]
                    
                    print('\n\n{} NEW EXPERIMENT {}'.format( '*' * 30, '*' * 30 ))
                    print('Carrying out experiment: {}/{}'.format(current_experiment_number, total_experiments))
                    print('Exceptions met: {}'.format(exceptions_met))
                    print('Model configuration:', my_model_configuration)
                    print('\n')
                    
                    configuration_variables = {
                        'results_path': results_path,
                        'versioning': versioning,
                        'reconduct_conducted_experiments': reconduct_conducted_experiments,
                        'count_continued_as_conducted': count_continued_as_conducted,
                        'save_continued': save_continued,
                        'threat_model': threat_model
                    }
                                
                    all_processes.append(
                        sub_main(
                            configuration_variables,
                            my_model_configuration,
                            my_backdoor_configuration
                        )
                    )
                
    mp_helper = Helper_Multiprocessing(all_processes, shots_at_a_time=shots_at_a_time, wait_before_starting_the_next_process=wait_before_starting_the_next_process)
    mp_helper.run_all_processes()
    
    return

