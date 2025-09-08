import torch
from termcolor import colored
import gc
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


# from ..config import *

from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset

from ..model_utils.torch_model_save_best import Torch_Model_Save_Best

from ..helper.data_helper import prepare_clean_and_poisoned_data, prepare_clean_and_poisoned_data_for_MR, prepare_clean_and_poisoned_data_for_MF, Limited_Dataset
from ..helper.defense_helper import get_defense
from ..helper.helper_class import Helper_Class

from ..snpca.npca_custom_with_masking import NPCA_Custom_with_Masking



def evaluation_shot_dc(
    configuration_variables: dict,
    my_model_configuration: dict,
    my_backdoor_configuration: dict,
    defense_configuration: dict,
    # my_evaluation_configuration: dict
):
    
    if defense_configuration['type']=='snpca_ood':
        return
    
    # *** preparing some results-related variables ***
    results_path = configuration_variables['results_path']
    reconduct_conducted_experiments = configuration_variables['reconduct_conducted_experiments']
    csv_file_path = results_path + my_model_configuration['dataset_name'] + '/csv_file/'
    force_overwrite = configuration_variables['force_overwrite_csv_results']
    
    # *** code starts here ***
    helper = Helper_Class(
        my_model_configuration=my_model_configuration,
        my_backdoor_configuration=my_backdoor_configuration,
        my_defense_configuration=defense_configuration
    )
    helper.prepare_paths_and_names(results_path, csv_file_path, model_name_prefix='central', filename='accuracies_and_losses_test.csv')
    
    my_data, poisoned_data = prepare_clean_and_poisoned_data(my_model_configuration, my_backdoor_configuration)
    helper.check_conducted(data_name=my_data.data_name, count_continued_as_conducted=False)
    
    if helper.experiment_conducted and (not reconduct_conducted_experiments):
        print('Experiment already conducted. Variable {reconduct_conducted_experiments} is set to False. Moving on to the next experiment.')
    
        global_model = Torch_Model_Save_Best(poisoned_data, my_model_configuration, path=helper.save_path)
        model_found = global_model.load_weights(global_model.save_directory + helper.model_name)
        if 'kaggle_imagenet' not in my_data.data_name:
            helper.test_on_clean_and_poisoned_data(global_model, my_data, poisoned_data)
        else:
            print('no test data used for ImageNet')
            # loaded = global_model.load_weights('__ignore__/ImageNetModels/PytorchModels/ResNet50/l2_improved_3_ep.pt')
            # loaded = global_model.load_weights('__ignore__/robust_resnet50_imagenet224.pth')
            # global_model.model.backbone_model = copy.deepcopy(clean_model.model.rn18)
            
        data_to_consider = poisoned_data
        data_to_evaluate = poisoned_data
        
        # make a copy of global model. we don't want to replace it.
        altered_model = Torch_Model_Save_Best(data_to_consider, my_model_configuration, path=helper.save_path)
        altered_model.load_weights(global_model.save_directory + helper.model_name)
        
        helper.evaluate(altered_model, data_to_consider, data_to_evaluate, defense_configuration=defense_configuration, scenario='DC', force_overwrite=force_overwrite, number_of_items=64)
        
    return


def evaluation_shot_mr(
    configuration_variables: dict,
    my_model_configuration: dict,
    my_backdoor_configuration: dict,
    defense_configuration: dict,
    # my_evaluation_configuration: dict
):
    
    if defense_configuration['type']=='snpca_id':
        return
    
    if defense_configuration['type'] == 'snpca_id':
        defense_configuration['type'] = 'snpca_ood'
    # -----------------------------
    
    # *** preparing some results-related variables ***
    num_evaluations = configuration_variables['num_evaluations']
    results_path = configuration_variables['results_path']
    reconduct_conducted_experiments = configuration_variables['reconduct_conducted_experiments']
    csv_file_path = results_path + my_model_configuration['dataset_name'] + '/csv_file/'
    force_overwrite = configuration_variables['force_overwrite_csv_results']
    
    # *** code starts here ***
    helper = Helper_Class(
        my_model_configuration=my_model_configuration,
        my_backdoor_configuration=my_backdoor_configuration,
        my_defense_configuration=defense_configuration,
        num_evaluations=num_evaluations
    )
    helper.prepare_paths_and_names(results_path, csv_file_path, model_name_prefix='central', filename='accuracies_and_losses_test.csv')
    
    my_data, poisoned_data, ood_data = prepare_clean_and_poisoned_data_for_MR(my_model_configuration, my_backdoor_configuration)
    helper.check_conducted(data_name=my_data.data_name, count_continued_as_conducted=False)
    
    if helper.experiment_conducted and (not reconduct_conducted_experiments):
        print('Experiment already conducted. Variable {reconduct_conducted_experiments} is set to False. Moving on to the next experiment.')
    
        global_model = Torch_Model_Save_Best(poisoned_data, my_model_configuration, path=helper.save_path)
        model_found = global_model.load_weights(global_model.save_directory + helper.model_name)
        if 'kaggle_imagenet' not in my_data.data_name:
            helper.test_on_clean_and_poisoned_data(global_model, my_data, poisoned_data)
        else:
            print('no test data used for ImageNet')
            # loaded = global_model.load_weights('__ignore__/ImageNetModels/PytorchModels/ResNet50/l2_improved_3_ep.pt')
            # loaded = global_model.load_weights('__ignore__/robust_resnet50_imagenet224.pth')
            # global_model.model.backbone_model = copy.deepcopy(clean_model.model.rn18)
            
        data_to_consider = ood_data
        data_to_evaluate = poisoned_data
        
        # limited_data = deepcopy(my_data)
        # indices = np.array([
        #     np.random.choice(np.where(np.array(my_data.test.targets)==target)[0], size=defense_configuration['num_target_class_samples'], replace=False)
        #     for target in range(my_data.num_classes)
        # ]).reshape(-1)
        # limited_data.test = Client_SubDataset(
        #     my_data.test,
        #     indices=indices
        # )
        # limited_data.train = limited_data.test
        
        limited_data = Limited_Dataset(my_data, size=defense_configuration['num_target_class_samples'])
        altered_model = Torch_Model_Save_Best(limited_data, my_model_configuration, path=helper.save_path)
        altered_model.load_weights(global_model.save_directory + helper.model_name)
        
        helper.evaluate(altered_model, data_to_consider, data_to_evaluate, defense_configuration=defense_configuration, scenario='MR', force_overwrite=force_overwrite, number_of_items=64)
        
    return


def evaluation_shot_mf(
    configuration_variables: dict,
    _my_model_configuration: dict,
    my_backdoor_configuration: dict,
    defense_configuration: dict,
    # my_evaluation_configuration: dict
):
    
    if defense_configuration['type']=='snpca_id':
        return
    
    # =============================
    # Manipulating some things related to the dataset variables for model reuse scenario
    # =============================
    my_model_configuration = {}
    for key in _my_model_configuration.keys():
        if key not in ['new_dataset_name', 'old_dataset_name', 'train_size']:
            my_model_configuration[key] = _my_model_configuration[key]
    my_model_configuration['dataset_name'] = _my_model_configuration['old_dataset_name']
    
    ood_model_configuration = my_model_configuration.copy()
    ood_model_configuration['dataset_name'] = _my_model_configuration['dataset_name']
    # -----------------------------
    
    # *** preparing some results-related variables ***
    results_path = configuration_variables['results_path']
    reconduct_conducted_experiments = configuration_variables['reconduct_conducted_experiments']
    csv_file_path = results_path + my_model_configuration['dataset_name'] + '/csv_file/'
    force_overwrite = configuration_variables['force_overwrite_csv_results']
    
    
    # *** code starts here ***
    # ===========================================
    # This helper is used to load the model because we want an already trained model to load.
    # ===========================================
    helper = Helper_Class(
        my_model_configuration=my_model_configuration,
        my_backdoor_configuration=my_backdoor_configuration,
        my_defense_configuration=defense_configuration
    )
    helper.prepare_paths_and_names(results_path, csv_file_path, model_name_prefix='central', filename='accuracies_and_losses_test.csv')
    
    # ==========================================
    # This helper saves the model after training it on the out of distribution dataset
    # ==========================================
    another_helper = Helper_Class(
        my_model_configuration=ood_model_configuration,
        my_backdoor_configuration=my_backdoor_configuration,
        my_defense_configuration=defense_configuration
    )
    another_helper.prepare_paths_and_names(results_path, csv_file_path, model_name_prefix='central', filename='accuracies_and_losses_test.csv')
    
    my_data, poisoned_data = prepare_clean_and_poisoned_data(my_model_configuration, my_backdoor_configuration)
    helper.check_conducted(data_name=my_data.data_name, count_continued_as_conducted=False)
    
    if helper.experiment_conducted and (not reconduct_conducted_experiments):
        print('Experiment already conducted. Variable {reconduct_conducted_experiments} is set to False. Moving on to the next experiment.')
    
        global_model = Torch_Model_Save_Best(my_data, my_model_configuration, path=helper.save_path)
        model_found = global_model.load_weights(global_model.save_directory + helper.model_name)
        if 'kaggle_imagenet' not in my_data.data_name:
            helper.test_on_clean_and_poisoned_data(global_model, my_data, poisoned_data)
        else:
            print('no test data used for ImageNet')
            # loaded = global_model.load_weights('__ignore__/robust_resnet50_imagenet224.pth')
            
        my_data, ood_data, ood_poisoned_data = prepare_clean_and_poisoned_data_for_MF(_my_model_configuration, my_backdoor_configuration)
        data_to_consider = ood_data
        data_to_evaluate = ood_poisoned_data
        another_helper.check_conducted(data_name=data_to_consider.data_name, count_continued_as_conducted=False)
        
        # make a copy of global model. we don't want to replace it.
        altered_model = Torch_Model_Save_Best(data_to_consider, my_model_configuration, path=helper.save_path)
        
        if another_helper.experiment_conducted and (not reconduct_conducted_experiments):
            altered_model.load_weights(altered_model.save_directory+another_helper.model_name)
        else:
            altered_model.model.load_state_dict(global_model.model.state_dict())
            
            num_of_layers_to_train = 15 # max(2, len(altered_model.get_children()) // 10)
            altered_model.freeze_last_n_layers(n=None)
            altered_model.unfreeze_last_n_layers(n=num_of_layers_to_train)
            altered_model.train(epochs=3)
            
            global_model_name = global_model.save_directory+helper.model_name
            altered_model_name = altered_model.save_directory+another_helper.model_name
            assert global_model_name!=altered_model_name, print(f'Global model name and altered model name should be different.\nGlobal model name is: {global_model_name}\nAltered model name is: {altered_model_name}')
            
            altered_model.save(another_helper.model_name)
            print('saved model at:', altered_model_name)
        
        helper.evaluate(altered_model, data_to_consider, data_to_evaluate, defense_configuration=defense_configuration, scenario='MF', force_overwrite=force_overwrite, number_of_items=64)
        
    return


