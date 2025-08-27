import os
import torch
import numpy as np
import pandas as pd
import copy
from termcolor import colored
import time


from utils_.general_utils import confirm_directory

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor

from ..snpca.npca_paper import NPCA_Paper
from .defense_helper import get_defense

from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy, get_outputs

from ..visual_utils.image_processing import normalize
from ..visual_utils.matplotlib_utils import show_images_in_grid



class Helper_Class:
    
    def __init__(
        self,
        my_model_configuration: dict,
        my_attack_configuration: dict=None,
        my_backdoor_configuration: dict=None,
        my_defense_configuration: dict=None,
        num_evaluations: int=1,
        verbose :bool=True,
        versioning :bool=True,
        **kwargs
    ):
        
        self.my_model_configuration = my_model_configuration
        self.my_attack_configuration = my_attack_configuration
        self.my_backdoor_configuration = my_backdoor_configuration
        self.my_defense_configuration = my_defense_configuration
        
        self.dictionary_to_save = {}
        self.dictionary_to_load = {}
        
        self.last_client_results = {}
        self.re_evaluated_on_non_patient_server = 0
        self.num_evaluations = num_evaluations
        
        self.verbose = verbose
        self.versioning = versioning
        
        return
    
    
    def print_out(self, *statement, end='\n', local_verbose: bool=False):
        if self.verbose or local_verbose:
            print(*statement, end=end)
        return
    
    
    def prepare_model_name(self, model_name_prefix='simple'):
        
        self.model_name = ''
        for key in self.my_backdoor_configuration.keys():
            self.model_name += f'_({key}-{self.my_backdoor_configuration[key]})'
        self.model_name += '/'
        
        # =========================
        # Cheating - so that new models can be trained with lesser epochs and older epochs remain stored.
        # =========================
        self.available_epochs = self.my_model_configuration['epochs']
        if isinstance(self.available_epochs, list):
            self.my_model_configuration['epochs'] = np.max(self.available_epochs)
        else:
            self.available_epochs = [self.available_epochs]
        self.do_epochs = np.min(self.available_epochs)
        # ==========================
        
        self.model_name += f'{model_name_prefix}'
        for key in self.my_model_configuration.keys():
            if key == 'gpu_number': self.model_name += '_(gpu_number-0)'
            # elif (key == 'epochs') and isinstance(self.available_epochs, list): self.model_name += f'_({key}-{self.my_attack_configuration['epochs'][0]})'
            elif (key == 'split_type') or (key == 'alpha'): pass
            else: self.model_name += f'_({key}-{self.my_model_configuration[key]})'
        
        return
    
    
    def prepare_csv_things(self, csv_file_path: str, filename: str=''):
        
        if csv_file_path[-1] != '/':
            csv_file_path += '/'
        self.csv_file_path = csv_file_path
        confirm_directory(self.csv_file_path)
        
        self.col_name_identifier = self.model_name
        self.col_name_identifier_with_defense = f'(model={self.model_name})_(defense=vanilla)'
        if self.my_defense_configuration is not None:
            if 'key' in self.my_defense_configuration.keys():
                self.col_name_identifier_with_defense = f'(model={self.model_name})_(defense={self.my_defense_configuration['key']})'
        
        self.csv_path_and_filename = '{}{}'.format(self.csv_file_path, filename)
        self.csv_path_and_filename_stats = '{}stats_{}'.format(self.csv_file_path, filename)
        if filename[-4:] != '.csv':
            self.csv_path_and_filename += '.csv'
            
        self.print_out('csv file path is: {}'.format(self.csv_path_and_filename_stats))
        
        return
    
    
    def prepare_paths_and_names(
        self, 
        results_path: str, 
        csv_file_path: str, 
        model_name_prefix: str='simple', 
        filename: str='_'
    ):
        
        self.save_path = results_path
        confirm_directory(self.save_path)
        
        self.prepare_model_name(model_name_prefix=model_name_prefix)
        self.prepare_csv_things(csv_file_path, filename)
        
        return
        
        
    def check_conducted(self, data_name: str='', count_continued_as_conducted: bool=True, local_verbose: bool=False):
        
        if data_name == '':
            data_name = self.my_model_configuration['dataset_name']
        
        
        hash_str = '#' * 50
        # check if experiment has already been conduction
        self.experiment_conducted = False
        save_model_path = f'{self.save_path}{data_name}/torch_models/{self.model_name}.pth'
        
        if os.path.isfile(save_model_path):
            self.print_out(f'{hash_str}\nHurray...! Model file found at: {save_model_path}\n{hash_str}', local_verbose=local_verbose)
            self.experiment_conducted = True
        elif data_name == 'kaggle_imagenet':
            self.print_out(f'\nHurray...! Experiment has not been conducted, but because the dataset is Imagenet, we don\'t need to conduct the experiments.')
            self.experiment_conducted = True
        else:
            self.print_out(f'{hash_str}\nModel file not found at: {save_model_path}.\n{hash_str}', local_verbose=local_verbose)
            print('WARNING: Experiment has not been conducted.')
            
        return
    
    
    def __save_dataframe(self, force_overwrite: bool=False):
        
        # get maximum length of the current dictionary
        max_len_of_dict = 0
        for key in self.dictionary_to_save.keys():
            if len(self.dictionary_to_save[key]) > max_len_of_dict:
                max_len_of_dict = len(self.dictionary_to_save[key])
        for key in self.dictionary_to_save.keys():
            self.dictionary_to_save[key] += [self.dictionary_to_save[key][-1]] * (max_len_of_dict-len(self.dictionary_to_save[key]))
        
        # load the df file
        if os.path.isfile(self.csv_path_and_filename):
            df = pd.read_csv(self.csv_path_and_filename)
        else:
            df = pd.DataFrame({'None': [-1]})
            
        # adjust the length of either the dataframe or the dictionary to match each other
        if len(df) > max_len_of_dict:
            diff_len = len(df) - max_len_of_dict
            for key in self.dictionary_to_save.keys():
                self.dictionary_to_save[key] += [self.dictionary_to_save[key][-1]] * diff_len
        elif len(df) < max_len_of_dict:
            diff_len = max_len_of_dict - len(df)
            for i in range(diff_len):
                df.loc[len(df)] = [-1. for column in df.columns]
        
        # copy dictionary to the dataframe
        for key in self.dictionary_to_save.keys():
            if (key not in df.columns) or (force_overwrite):
                if key in df.columns:
                    self.print_out('Overwriting due to force overwrite.')
                assert len(df) == len(self.dictionary_to_save[key]), f'Length of dataframe is {len(df)}, but the length of array is {len(self.dictionary_to_save[key])}'
                df[key] = self.dictionary_to_save[key]
                
        # save the dataframe
        df.to_csv(self.csv_path_and_filename, index=False)
        
        return
    
    
    def save_dataframe(self, force_overwrite: bool=False):
        
        # load the df file
        if os.path.isfile(self.csv_path_and_filename_stats):
            df = pd.read_csv(self.csv_path_and_filename_stats)
        else:
            df = pd.DataFrame({'None': [-1]})
            
        # adjust the length of either the dataframe or the dictionary to match each other
        assert len(df) == 1, 'The lenght of the dataframe should be 1.'
        
        # copy dictionary to the dataframe
        for key in self.dictionary_to_save.keys():
            if (key not in df.columns) or (force_overwrite):
                if key in df.columns:
                    self.print_out('Overwriting due to force overwrite.')
                df[key] = self.dictionary_to_save[key]
                
        # save the dataframe
        df.to_csv(self.csv_path_and_filename_stats, index=False)
        
        return
    
    
    def load_columns_in_dictionary(self, load_columns, local_verbose: bool=False):
        
        # load the df file
        try:
            df = pd.read_csv(self.csv_path_and_filename_stats)
            
            for column in load_columns:
                if column in df.columns:
                    self.dictionary_to_load[column] = df[column].tolist()
        
        except:
            if local_verbose: 
                print(f'No experiments for the data/model configuration: {self.my_model_configuration['dataset_name']} have been found.')
            pass
        
        return
    
    
    def check_columns(self, load_columns: list[str], local_verbose: bool=False):
        
        # load the df file
        try:
            df = pd.read_csv(self.csv_path_and_filename_stats)
            return [column in df.columns for column in load_columns]
        
        except:
            if local_verbose: 
                print(f'No experiments for the data/model configuration: {self.my_model_configuration['dataset_name']} have been found.')
            pass
        
        return [False for column in load_columns]
    
    
    def evaluate(
        self, 
        altered_model: Torch_Model, 
        data_to_consider: Torch_Dataset, data_to_evaluate: Simple_Backdoor, 
        defense_configuration: dict={}, scenario: str='DC', 
        force_overwrite: bool=False,
        **kwargs
    ):
        
        keys = ['cl', 'ca', 'pl', 'pa']
        scenario_str = f'_{scenario}' if scenario!='DC' else ''
        all_keys = [f'{self.col_name_identifier_with_defense}_{key}{scenario_str}' for key in keys]
        keys_present = True if np.sum(np.array(self.check_columns(all_keys)).astype(np.float32))==len(all_keys) else False
        if keys_present and (not force_overwrite):
            print(colored('All keys already found. Moving on to the next experiment.', 'yellow'))
            return
        else:
            print(colored('Going to perform experiments.', 'yellow'))
            
        
        iterations = self.num_evaluations if 'snpca_ood' in defense_configuration['type'] else 1
        
        losses_c, accs_c, losses_p, accs_p = [], [], [], []
        for i in range(iterations):
            defense = get_defense(data_to_consider, altered_model, defense_configuration=defense_configuration)
            defense.defend()
            (loss_clean, acc_clean), (loss_poisoned, acc_poisoned) = self.evaluate_without_saving(defense, data_to_evaluate, **kwargs)
            losses_c.append(loss_clean); accs_c.append(acc_clean);
            losses_p.append(loss_poisoned); accs_p.append(acc_poisoned);
            
            scenario_str = f'_{scenario}' if scenario!='DC' else ''
            scenario_str += f'_(iteration={i})'
            self.dictionary_to_save = {
                f'{self.col_name_identifier_with_defense}_cl{scenario_str}': [loss_clean],
                f'{self.col_name_identifier_with_defense}_ca{scenario_str}': [acc_clean],
                f'{self.col_name_identifier_with_defense}_pl{scenario_str}': [loss_poisoned],
                f'{self.col_name_identifier_with_defense}_pa{scenario_str}': [acc_poisoned],
            }
            
        loss_clean = np.median(losses_c); acc_clean = np.median(accs_c);
        loss_poisoned = np.median(losses_p); acc_poisoned = np.median(accs_p);
        
        scenario_str = f'_{scenario}' if scenario!='DC' else ''
        self.dictionary_to_save = {
            f'{self.col_name_identifier_with_defense}_cl{scenario_str}': [loss_clean],
            f'{self.col_name_identifier_with_defense}_ca{scenario_str}': [acc_clean],
            f'{self.col_name_identifier_with_defense}_pl{scenario_str}': [loss_poisoned],
            f'{self.col_name_identifier_with_defense}_pa{scenario_str}': [acc_poisoned],
        }
        
        self.save_dataframe(force_overwrite=force_overwrite)
        
        return
        
        
    def evaluate_defense(self, defense: NPCA_Paper, data_to_consider: Simple_Backdoor, scenario: str='DC', **kwargs):
        
        # if self.my_defense_configuration['key'] != 'vanilla':
        #     (loss_clean, acc_clean), (loss_poisoned, acc_poisoned) = defense.evaluate(data_to_consider)
        # else:
        #     model = defense.torch_model
        #     loss_clean, acc_clean = model.test_shot(torch.utils.data.DataLoader(data_to_consider.test, batch_size=model.model_configuration['batch_size']))
        #     loss_poisoned, acc_poisoned = model.test_shot(torch.utils.data.DataLoader(data_to_consider.poisoned_test, batch_size=model.model_configuration['batch_size']))
        #     print()
        
        (loss_clean, acc_clean), (loss_poisoned, acc_poisoned) = self.evaluate_without_saving(defense, data_to_consider, **kwargs)
        
        scenario_str = f'_{scenario}' if scenario!='DC' else ''
        self.dictionary_to_save = {
            f'{self.col_name_identifier_with_defense}_cl{scenario_str}': [loss_clean],
            f'{self.col_name_identifier_with_defense}_ca{scenario_str}': [acc_clean],
            f'{self.col_name_identifier_with_defense}_pl{scenario_str}': [loss_poisoned],
            f'{self.col_name_identifier_with_defense}_pa{scenario_str}': [acc_poisoned],
        }
        
        self.save_dataframe(force_overwrite=True)
        
        return
    
    
    def evaluate_without_saving(self, defense: NPCA_Paper, data_to_consider: Simple_Backdoor, **kwargs):
        
        new_data_to_consider = copy.deepcopy(data_to_consider)
        batch_size = defense.torch_model.model_configuration['batch_size']
        
        clean_dl = torch.utils.data.DataLoader(new_data_to_consider.test, batch_size=batch_size)
        x, y = get_data_samples_from_loader(clean_dl, return_numpy=True)
        y_predicted = np.argmax(get_outputs(defense.torch_model.model, clean_dl, return_numpy=True), axis=1)
        cleanly_labelled_indices = np.where((y==y_predicted) & (y!=data_to_consider.backdoor_configuration['target']))[0]
        
        new_data_to_consider.test = Client_SubDataset(data_to_consider.test, cleanly_labelled_indices)
        new_data_to_consider.poisoned_test = Client_SubDataset(data_to_consider.poisoned_test, cleanly_labelled_indices)
        
        # if self.my_defense_configuration['key'] != 'vanilla':
        #     (loss_clean, acc_clean), (loss_poisoned, acc_poisoned) = defense.evaluate(data_to_consider)
        # else:
        #     model = defense.torch_model
        #     loss_clean, acc_clean = model.test_shot(torch.utils.data.DataLoader(data_to_consider.test, batch_size=model.model_configuration['batch_size']))
        #     loss_poisoned, acc_poisoned = model.test_shot(torch.utils.data.DataLoader(new_data_to_consider.poisoned_test, batch_size=model.model_configuration['batch_size']))
        #     print()
        # losses_c, accs_c, losses_p, accs_p = [], [], [], []
        # for i in range(2):
        #     (loss_clean, acc_clean), (loss_poisoned, acc_poisoned) = defense.evaluate(data_to_consider, **kwargs)
        #     losses_c.append(loss_clean); accs_c.append(acc_clean);
        #     losses_p.append(loss_poisoned); accs_p.append(acc_poisoned);
        # loss_clean = np.mean(losses_c); acc_clean = np.mean(accs_c);
        # loss_poisoned = np.mean(losses_p); acc_poisoned = np.mean(accs_p);
        
        (loss_clean, acc_clean), (loss_poisoned, acc_poisoned) = defense.evaluate(data_to_consider, **kwargs)
        
        return (loss_clean, acc_clean), (loss_poisoned, acc_poisoned)
    
    
    def test_on_clean_and_poisoned_data(self, global_model: Torch_Model, my_data: Torch_Dataset, poisoned_data: Simple_Backdoor):
        
        # prepare torch dataloaders from data
        clean_data_loader = torch.utils.data.DataLoader(my_data.test, batch_size=self.my_model_configuration['batch_size'], shuffle=True)
        poisoned_data_loader = torch.utils.data.DataLoader(poisoned_data.poisoned_test, batch_size=self.my_model_configuration['batch_size'], shuffle=True)
        
        # test the model on clean data
        global_model.test_shot(clean_data_loader)
        print()
        global_model.test_shot(poisoned_data_loader)
        
        return
    
    
    def test_model(self, _mnpca: NPCA_Paper, my_data: Torch_Dataset, poisoned_data: Simple_Backdoor):
        
        _, data_loader_target_class = _mnpca.get_data_subset_personal(my_data.test, _mnpca.target_class, bs=_mnpca.model.model_configuration['batch_size'])
        
        # before purification, compute the accuracy of the model
        if 'kaggle_imagenet' not in my_data.data_name:
            self.test_on_clean_and_poisoned_data(_mnpca.original_model, my_data, poisoned_data)
        print(f'\nAnd accuracy of target class is: {_mnpca.model.test_shot(data_loader_target_class, verbose=False)[1]}.\n')
        
        return
    
    
    