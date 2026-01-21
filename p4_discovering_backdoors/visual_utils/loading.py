import numpy as np
import copy


from ..config import *
from ..helper.helper_class import Helper_Class



dataset_name_correction = {
    'gtsrb': 'gtsrb',
    'cifar10': 'cifar10',
    'mnist': 'mnist'
}


def _load_results_from_settings(
    dataset_names, 
    backdoor_types, 
    defense_types,
    keys: list[str] = ['ca'],
    continued = True,
    results_path_local :str = '',
    verbose: bool = False
):
    
    _results_path = results_path_local
    
    results_arr = np.zeros((
        len(dataset_names),
        len(backdoor_types),
        len(defense_types),
        len(keys)
    ))
    
    total_results_to_load = len(dataset_names) * len(backdoor_types) * len(defense_types)
    load_results_number = 0
    
    # iterating over data
    for d_ind, dataset_name in enumerate(dataset_names):
        for key in dataset_name_correction.keys():
            if key in dataset_name:
                correct_name = dataset_name_correction[key]
        
        # iterating over clients distibutions
        for b, backdoor_type in enumerate(backdoor_types):
            
            # iterating over server configurations
            for d, defense_type in enumerate(defense_types):
                if verbose: print(f'\rLoading results {load_results_number}/{total_results_to_load}', end='')
                
                my_model_configuration = copy.deepcopy(model_configurations[dataset_name])
                my_model_configuration['dataset_name'] = dataset_name
                
                my_backdoor_configuration = copy.deepcopy(all_backdoor_configurations[configured_backdoors[backdoor_type]['type']])
                for key in configured_backdoors[backdoor_type].keys():
                    my_backdoor_configuration[key] = configured_backdoors[backdoor_type][key]
                    
                my_defense_configuration = copy.deepcopy(all_defense_configurations[configured_defenses[defense_type]['type']])
                for key in configured_defenses[defense_type].keys():
                    my_defense_configuration[key] = configured_defenses[defense_type][key]
                my_defense_configuration['key'] = defense_type
                
                # *** preparing some results-related variables ***
                csv_file_path = _results_path + dataset_name + '/csv_file/'
                helper = Helper_Class(
                    my_model_configuration=my_model_configuration,
                    my_backdoor_configuration=my_backdoor_configuration,
                    my_defense_configuration=my_defense_configuration,
                    verbose=False, versioning=False
                )
                helper.prepare_paths_and_names(_results_path, csv_file_path, model_name_prefix='central', filename='accuracies_and_losses_test.csv')
                # helper.check_conducted(local_verbose=True)
                
                load_columns = [helper.col_name_identifier_with_defense + '_' + key for key in keys]
                print(load_columns)
                helper.load_columns_in_dictionary(load_columns)
                for k, key in enumerate(keys):
                    load_column = helper.col_name_identifier_with_defense + '_' + key
                    if load_column in helper.dictionary_to_load.keys():
                        
                    #     if continued:
                    #         for i in range(1, len(helper.dictionary_to_load[load_column])):
                    #             if helper.dictionary_to_load[load_column][i] == -1:
                    #                 helper.dictionary_to_load[load_column][i] = helper.dictionary_to_load[load_column][i-1]
                        
                        results_arr[d_ind, b, d, k] = helper.dictionary_to_load[load_column][-1]
                        
                    else:
                        results_arr[d_ind, b, d, k] = -2.
                
                load_results_number += 1
                
    return results_arr


def load_results_from_settings(
    dataset_names, 
    backdoor_types, 
    defense_types,
    keys: list[str] = ['ca'],
    continued = True,
    results_path_local :str = '',
    verbose: bool = False
):
    
    _results_path = results_path_local
    
    results_arr = np.zeros((
        len(dataset_names),
        len(backdoor_types),
        len(defense_types),
        len(keys)
    ))
    
    total_results_to_load = len(dataset_names) * len(backdoor_types) * len(defense_types)
    load_results_number = 0
    
    # iterating over data
    for d_ind, dataset_name in enumerate(dataset_names):
        
        # iterating over clients distibutions
        for b, backdoor_type in enumerate(backdoor_types):
            
            # iterating over server configurations
            for d, defense_type in enumerate(defense_types):
                if verbose: print(f'\rLoading results {load_results_number}/{total_results_to_load}', end='')
                
                _my_model_configuration = copy.deepcopy(model_configurations[dataset_name])
                _my_model_configuration['dataset_name'] = dataset_name
                
                if 'old_dataset_name' in _my_model_configuration.keys():
                    my_new_model_configuration = {}
                    for key in _my_model_configuration.keys():
                        if key not in ['new_dataset_name', 'old_dataset_name', 'train_size']:
                            my_new_model_configuration[key] = _my_model_configuration[key]
                    my_new_model_configuration['dataset_name'] = _my_model_configuration['old_dataset_name']
                    
                    my_model_configuration = copy.deepcopy(my_new_model_configuration)
                    my_model_configuration['dataset_name'] = _my_model_configuration['new_dataset_name']
                else:
                    my_model_configuration = copy.deepcopy(_my_model_configuration)
                
                
                my_backdoor_configuration = copy.deepcopy(all_backdoor_configurations[configured_backdoors[backdoor_type]['type']])
                for key in configured_backdoors[backdoor_type].keys():
                    my_backdoor_configuration[key] = configured_backdoors[backdoor_type][key]
                    
                my_defense_configuration = copy.deepcopy(all_defense_configurations[configured_defenses[defense_type]['type']])
                for key in configured_defenses[defense_type].keys():
                    my_defense_configuration[key] = configured_defenses[defense_type][key]
                my_defense_configuration['key'] = defense_type
                
                # *** preparing some results-related variables ***
                csv_file_path = _results_path + my_model_configuration['dataset_name'] + '/csv_file/'
                helper = Helper_Class(
                    my_model_configuration=my_model_configuration,
                    my_backdoor_configuration=my_backdoor_configuration,
                    my_defense_configuration=my_defense_configuration,
                    verbose=False, versioning=False
                )
                helper.prepare_paths_and_names(_results_path, csv_file_path, model_name_prefix='central', filename='accuracies_and_losses_test.csv')
                # helper.check_conducted(local_verbose=True)
                
                load_columns = [helper.col_name_identifier_with_defense + '_' + key for key in keys]
                helper.load_columns_in_dictionary(load_columns)
                for k, key in enumerate(keys):
                    load_column = helper.col_name_identifier_with_defense + '_' + key
                    if load_column in helper.dictionary_to_load.keys():
                        
                    #     if continued:
                    #         for i in range(1, len(helper.dictionary_to_load[load_column])):
                    #             if helper.dictionary_to_load[load_column][i] == -1:
                    #                 helper.dictionary_to_load[load_column][i] = helper.dictionary_to_load[load_column][i-1]
                        
                        results_arr[d_ind, b, d, k] = helper.dictionary_to_load[load_column][-1]
                        
                    else:
                        print(f'Column {load_column} not found in {helper.csv_path_and_filename_stats}.')
                        results_arr[d_ind, b, d, k] = -2.
                
                load_results_number += 1
                
    return results_arr

