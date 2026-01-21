import torch
from termcolor import colored
import gc


# from ..config import *

from ..model_utils.torch_model_save_best import Torch_Model_Save_Best

from ..helper.data_helper import prepare_clean_and_poisoned_data
from ..helper.helper_class import Helper_Class



def poisoned_training_shot(
    configuration_variables: dict,
    my_model_configuration: dict,
    my_backdoor_configuration: dict
):
    
    # *** preparing some results-related variables ***
    results_path = configuration_variables['results_path']
    reconduct_conducted_experiments = configuration_variables['reconduct_conducted_experiments']
    csv_file_path = results_path + my_model_configuration['dataset_name'] + '/csv_file/'
    threat_model = configuration_variables['threat_model']
    
    
    # *** Training code starts here ***
    my_data, poisoned_data = prepare_clean_and_poisoned_data(my_model_configuration, my_backdoor_configuration)
    
    helper = Helper_Class(my_model_configuration=my_model_configuration, my_backdoor_configuration=my_backdoor_configuration)
    helper.prepare_paths_and_names(results_path, csv_file_path, model_name_prefix='central', filename='accuracies_and_losses_test.csv')
    helper.model_name = f'{threat_model}/' + helper.model_name if (threat_model=='MR') and (poisoned_data.requires_training_control) else helper.model_name
    
    helper.check_conducted(data_name=my_data.data_name, count_continued_as_conducted=False)
    
    if helper.experiment_conducted and (not reconduct_conducted_experiments):
        print('Experiment already conducted. Variable {reconduct_conducted_experiments} is set to False. Moving on to the next experiment.')
    
    else:
        global_model = Torch_Model_Save_Best(poisoned_data, my_model_configuration, path=helper.save_path)
        
        # Freezing initial layers for better transfer learning on the transformer-based models
        # Only for transformer-based models because they typically require more data to train otherwise
        suffixes_of_transformer_models = ['vit16', 'convnext']
        for suffix in suffixes_of_transformer_models:
            if suffix in helper.my_model_configuration['model_architecture']:
                global_model.unfreeze_last_n_layers(n=None)
                global_model.freeze_last_n_layers(n=None)
                global_model.unfreeze_last_n_layers(n=10)
        
        global_model.train(
            start_epoch=1, 
            epochs=helper.do_epochs, batch_size=helper.my_model_configuration['batch_size'], 
            save_best_model=True, shuffle=True, training_type=threat_model
        )
        print()
        
        global_model.save(helper.model_name)
        print('saved model at:', global_model.save_directory + helper.model_name)
        
        test_loader = torch.utils.data.DataLoader(my_data.test, shuffle=True, batch_size=helper.my_model_configuration['batch_size'])
        global_model.test_shot(test_loader, verbose=True)
        print()
    
    return

