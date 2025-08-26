from p4_discovering_backdoors.config import visible_gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu

import argparse


from p4_discovering_backdoors.training import main as main_test

from p4_discovering_backdoors.scripts.train_poisoned_multi import main as main_train
from p4_discovering_backdoors.scripts.evaluation_multi import main as main_evaluation
from p4_discovering_backdoors.scripts.results import main as main_results
# from p4_discovering_backdoors.scripts.evaluate_backdoor_multi import main as evaluate_backdoor



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='SNPCA')
    parser.add_argument('--train', action='store_true', default=False, help='Trains new models if not found in the directory.')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Evaluates all the defenses in the config.')
    parser.add_argument('--threat_model', type=str, default='DC', choices=['DC', 'MR', 'MF', 'DCMR', 'all'], help='The threat model to use.')
    parser.add_argument('--results', action='store_true', default=False, help='Prints tables in the terminal in latex format and saves results figures.')
    all_arguments = parser.parse_args()
    
    
    if all_arguments.train:
        main_train()
    
    if all_arguments.evaluate:
        threat_models = ['DC', 'MR', 'MF'] if all_arguments.threat_model=='all' else [all_arguments.threat_model]
        threat_models = ['DC', 'MR'] if all_arguments.threat_model=='DCMR' else threat_models
        [main_evaluation(scenario=threat_model) for threat_model in threat_models]
        # main_evaluation(scenario='DC')
        # main_evaluation(scenario='MR')
        # main_evaluation(scenario='MF')
    
    if all_arguments.results:
        threat_models = ['DC', 'MR', 'MF'] if all_arguments.threat_model=='all' else [all_arguments.threat_model]
        threat_models = ['DC', 'MR'] if all_arguments.threat_model=='DCMR' else threat_models
        [main_results(scenario=threat_model) for threat_model in threat_models]
    
    print()
    
    