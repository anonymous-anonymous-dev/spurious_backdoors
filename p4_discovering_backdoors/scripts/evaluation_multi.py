import numpy as np
import gc
from copy import deepcopy

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


from ..config import *

from .evaluate_shot import evaluation_shot_dc, evaluation_shot_mr, evaluation_shot_mf

from ..helper.helper_multiprocessing import Helper_Multiprocessing



def sub_main_mp(
    kwargs
):
    
    # *** preparing variables out of kwargs ***
    scenario = kwargs['scenario']
    configuration_variables = kwargs['configuration_variables']
    my_model_configuration = kwargs['my_model_configuration']
    my_backdoor_configuration = kwargs['my_backdoor_configuration']
    my_defense_configuration = kwargs['my_defense_configuration']
    
    if scenario == 'DC':
        evaluation_shot_dc(
            configuration_variables,
            my_model_configuration,
            my_backdoor_configuration,
            my_defense_configuration
        )
        
    elif scenario == 'MR':
        evaluation_shot_mr(
            configuration_variables,
            my_model_configuration,
            my_backdoor_configuration,
            my_defense_configuration
        )
        
    elif scenario == 'MF': 
        evaluation_shot_mf(
            configuration_variables,
            my_model_configuration,
            my_backdoor_configuration,
            my_defense_configuration
        )
    
    return


def sub_main(
    scenario: str,
    configuration_variables: dict,
    my_model_configuration: dict,
    my_backdoor_configuration: dict,
    my_defense_configuration: dict
):
    
    return multiprocessing.Process(
        target = sub_main_mp,
        args = (
            {
                'scenario': scenario,
                'configuration_variables': configuration_variables,
                'my_model_configuration': my_model_configuration,
                'my_backdoor_configuration': my_backdoor_configuration,
                'my_defense_configuration': my_defense_configuration
            },
        )
    )


def main(orientation=0, scenario='DC'):
    
    if orientation == 1:
        _experimental_setups = experimental_setups[::-1]
    else:
        _experimental_setups = experimental_setups
    
    # starts here
    current_experiment_number = 0; exceptions_met = 0
    all_processes = []
    for experimental_setup in _experimental_setups:
        dataset_names = experimental_setup.dataset_names
        # if scenario == 'TM2':
        #     dataset_names = experimental_setup.tm2_dataset_names
        if scenario == 'MF':
            dataset_names = experimental_setup.mf_dataset_names
        backdoor_attack_types = experimental_setup.backdoor_attack_types
        defense_types = experimental_setup.defense_types
        
        total_experiments = len(dataset_names) * len(backdoor_attack_types) * len(defense_types)
    
        # setting the orientation of the experiment
        if orientation == 1:
            _dataset_names = dataset_names[::-1]
        else:
            _dataset_names = dataset_names
        
        # iterating over data
        for dataset_name in _dataset_names:
            
            # iterating over backdoor attacks
            for backdoor_type in backdoor_attack_types:
                
                # iterating over evaluation configs
                for defense_type in defense_types:
                    
                    my_model_configuration = deepcopy(model_configurations[dataset_name])
                    my_model_configuration['dataset_name'] = dataset_name
                    
                    my_backdoor_configuration = deepcopy(all_backdoor_configurations[configured_backdoors[backdoor_type]['type']])
                    for key in configured_backdoors[backdoor_type].keys():
                        my_backdoor_configuration[key] = configured_backdoors[backdoor_type][key]
                        
                    my_defense_configuration = deepcopy(all_defense_configurations[configured_defenses[defense_type]['type']])
                    for key in configured_defenses[defense_type].keys():
                        if isinstance(configured_defenses[defense_type][key], dict):
                            new_dict = {} if key not in my_defense_configuration.keys() else deepcopy(my_defense_configuration[key])
                            for _key in configured_defenses[defense_type][key].keys():
                                new_dict[_key] = configured_defenses[defense_type][key][_key]
                            my_defense_configuration[key] = deepcopy(new_dict)
                        else:
                            my_defense_configuration[key] = configured_defenses[defense_type][key]
                    my_defense_configuration['key'] = defense_type
                    
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
                        'force_overwrite_csv_results': force_overwrite_csv_results,
                        'num_evaluations': num_evaluations
                    }
                                
                    all_processes.append(
                        sub_main(
                            scenario,
                            configuration_variables,
                            my_model_configuration,
                            my_backdoor_configuration,
                            my_defense_configuration
                        )
                    )
                
    mp_helper = Helper_Multiprocessing(all_processes, shots_at_a_time=shots_at_a_time)
    mp_helper.run_all_processes()
    
    return

