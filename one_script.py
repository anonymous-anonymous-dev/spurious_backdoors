import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from __notebooks__.p4.test_new_pca import see_results
from copy import deepcopy
from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.datasets import GTSRB, Kaggle_Imagenet, MNIST, MNIST_3
import numpy as np
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model
from termcolor import colored
import time

from _1_adversarial_ML.adversarial_attacks.pgd import PGD
from _1_adversarial_ML.adversarial_attacks.fgsm import FGSM
from _1_adversarial_ML.adversarial_attacks.all_available_adversarial_attacks import i_FGSM
from _1_adversarial_ML.adversarial_attacks.universal_adversarial_perturbation import Universal_Adversarial_Perturbation
from _1_adversarial_ML.backdoor_attacks.backdoor_data import Simple_Backdoor, Invisible_Backdoor

from _1_adversarial_ML.backdoor_defenses.post_training_defenses.strip import STRIP
from _1_adversarial_ML.backdoor_defenses.post_training_defenses.spectral_signatures import Spectral_Clustering
from _1_adversarial_ML.backdoor_defenses.post_training_defenses.activation_clustering import Activation_Clustering

from p4_discovering_backdoors.snpca.npca_paper import NPCA_Paper
from p4_discovering_backdoors.snpca.npca_custom_no_masking import NPCA_Custom
from p4_discovering_backdoors.attacks.input_minimalist_patch import Patch_Input_Minimalist
from p4_discovering_backdoors.model_utils.feature_activations import Feature_Activations
from p4_discovering_backdoors.snpca.snpca_analysis import SNPCA_Analysis
from p4_discovering_backdoors.snpca.analyzer import PCA_Analyzer

import torch, numpy as np
from utils_.general_utils import normalize, exponential_normalize
from utils_.visual_utils import show_image_grid
from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy, get_outputs

from utils_.pca import PCA, PCA_of_SKLEARN, PCA_SKLEARN_MEDIAN, Sparse_PCA_of_SKLEARN

import importlib
from p4_discovering_backdoors.model_utils import quantizer
importlib.reload(quantizer)
from p4_discovering_backdoors.model_utils.quantizer import Quantization

from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from __notebooks__.p4.test_new_pca import see_results

import matplotlib.pyplot as plt
import numpy as np
from __notebooks__.p4.test_new_pca import see_results
from copy import deepcopy
from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.datasets import GTSRB, Kaggle_Imagenet, MNIST, MNIST_3, CIFAR100, CIFAR10
import numpy as np
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model
from _0_general_ML.model_utils.activations_wrapper import Feature_Activations
from _0_general_ML.model_utils.generalized_model_activations_wrapper import Dependable_Feature_Activations

from _1_adversarial_ML.adversarial_attacks.pgd import PGD
from _1_adversarial_ML.adversarial_attacks.fgsm import FGSM
from _1_adversarial_ML.adversarial_attacks.all_available_adversarial_attacks import i_FGSM
from _1_adversarial_ML.adversarial_attacks.universal_adversarial_perturbation import Universal_Adversarial_Perturbation
from _1_adversarial_ML.backdoor_attacks.backdoor_data import Simple_Backdoor, Invisible_Backdoor

from _1_adversarial_ML.backdoor_defenses.post_training_defenses.strip import STRIP
from _1_adversarial_ML.backdoor_defenses.post_training_defenses.spectral_signatures import Spectral_Clustering
from _1_adversarial_ML.backdoor_defenses.post_training_defenses.activation_clustering import Activation_Clustering

from p4_discovering_backdoors.snpca.npca_paper import NPCA_Paper
from p4_discovering_backdoors.snpca.npca_custom_no_masking import NPCA_Custom
from p4_discovering_backdoors.snpca.npca_random import NPCA_Random
from p4_discovering_backdoors.attacks.input_minimalist_patch import Patch_Input_Minimalist
from p4_discovering_backdoors.model_utils.feature_activations import Feature_Activations
from p4_discovering_backdoors.snpca.snpca_analysis import SNPCA_Analysis
# from p4_discovering_backdoors.snpca.analyzer import PCA_Analyzer
from p4_discovering_backdoors.snpca.ood_trainer_analyzer import Adversarially_Smoothed_NPCA
from p4_discovering_backdoors.snpca.snpca_defense import Adversarially_Smoothed_NPCA_Defense
from p4_discovering_backdoors.snpca.one_channel_data import Channel1_Torch_Dataset
from p4_discovering_backdoors.model_utils.wrapping_utils import get_wrapped_model
from p4_discovering_backdoors.helper.defense_helper import get_defense

from p4_discovering_backdoors.attacks.input_minimalist import Input_Minimalist
from p4_discovering_backdoors.attacks.input_minimalist_patch import Patch_Input_Minimalist
from p4_discovering_backdoors.attacks.fgsm_attack import FGSM_with_Dict
from p4_discovering_backdoors.attacks.adv_attack import Random_Patch_Adversarial_Attack

import torch, numpy as np
from utils_.general_utils import normalize, exponential_normalize
from utils_.visual_utils import show_image_grid
from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy, get_outputs

from utils_.pca import PCA, PCA_of_SKLEARN, PCA_SKLEARN_MEDIAN, Sparse_PCA_of_SKLEARN
import math

import importlib
from p4_discovering_backdoors.model_utils import quantizer
importlib.reload(quantizer)
from p4_discovering_backdoors.model_utils.quantizer import Quantization

from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from __notebooks__.p4.test_new_pca import see_results

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from p4_discovering_backdoors.model_utils.torch_model_save_best import Torch_Model_Save_Best

from p4_discovering_backdoors.helper.data_helper import prepare_clean_and_poisoned_data
from p4_discovering_backdoors.helper.helper_class import Helper_Class

from p4_discovering_backdoors.config import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_pdf import PdfPages


def test_on_clean_and_poisoned_data(global_model: Torch_Model_Save_Best, my_data, poisoned_data, helper):
    # prepare torch dataloaders from data
    clean_data_loader = torch.utils.data.DataLoader(my_data.test, batch_size=helper.my_model_configuration['batch_size'], shuffle=False)
    poisoned_data_loader = torch.utils.data.DataLoader(poisoned_data.poisoned_test, batch_size=helper.my_model_configuration['batch_size'], shuffle=False)
    
    # test the model on clean data
    global_model.test_shot(clean_data_loader)
    print()
    global_model.test_shot(poisoned_data_loader)
    
    return

class Custom_Dataset(Torch_Dataset):
    def __init__(self, ood_data: Torch_Dataset, train_size=None, max_target=9):
        super().__init__(ood_data.data_name, ood_data.preferred_size, ood_data.data_means, ood_data.data_stds)
        targets = np.where(np.array(ood_data.train.targets) <= max_target)[0]
        train_size = train_size if train_size is not None else len(targets)
        self.train = Client_SubDataset(ood_data.train, indices=np.random.choice(targets, size=train_size, replace=False))
        test_targets = np.where(np.array(ood_data.test.targets) <= max_target)[0]
        self.test = Client_SubDataset(ood_data.test, indices=test_targets)
        self.num_classes = max_target+1
        return

# *** setting up the experiment ***
experiment_folder = 'results_snpca_1/'
dataset_name = 'cifar100'
# dataset_name = 'kaggle_imagenet_vit_b_16'
model_name = None #'cifar10_vit16'
backdoor_attack_type = 'invisible_backdoor'
# backdoor_attack_type = 'reflection_backdoor'
backdoor_attack_type = 'simple_backdoor'
# backdoor_attack_type = 'clean_label_backdoor'
# backdoor_attack_type = 'horizontal_backdoor'
target_class = 0
poisoning_ratio = 0.

# *** preparing some results-related variables ***
results_path = '../../__all_results__/_p4_discovering_backdoors/' + experiment_folder
my_model_configuration = model_configurations[dataset_name if model_name is None else model_name]
my_model_configuration['dataset_name'] = dataset_name
csv_file_path = results_path + my_model_configuration['dataset_name'] + '/csv_file/'

new_backdoor_configuration = {'poison_ratio': poisoning_ratio, 'poison_ratio_wrt_class_members': True, 'type': backdoor_attack_type}
my_backdoor_configuration = all_backdoor_configurations[configured_backdoors[backdoor_attack_type]['type']].copy()
# for key in configured_backdoors[backdoor_attack_type].keys():
#     my_backdoor_configuration[key] = configured_backdoors[backdoor_attack_type][key]
for key in new_backdoor_configuration.keys():
    my_backdoor_configuration[key] = new_backdoor_configuration[key]

helper = Helper_Class(my_model_configuration=my_model_configuration, my_backdoor_configuration=my_backdoor_configuration)
helper.prepare_paths_and_names(results_path, csv_file_path, model_name_prefix='central', filename='accuracies_and_losses_test.csv')

my_data, poisoned_data = prepare_clean_and_poisoned_data(my_model_configuration, my_backdoor_configuration)
helper.check_conducted(data_name=my_data.data_name, count_continued_as_conducted=False)

global_model = Torch_Model_Save_Best(poisoned_data, my_model_configuration, path=helper.save_path)
# global_model.unfreeze_last_n_layers(n=None)
global_model.freeze_last_n_layers(n=None)
# global_model.unfreeze_last_n_layers(n=8)
# restore the best test model
model_found = global_model.load_weights(global_model.save_directory + helper.model_name)
if (not model_found) and ('imagenet' not in my_data.data_name):
    global_model.train(epochs=50)
    print(colored('Model not found.'))
else:
    if 'imagenet' not in my_data.data_name:
        test_on_clean_and_poisoned_data(global_model, my_data, poisoned_data, helper); print();
    else:
        print('no test data used for ImageNet')
        
        
defense_type = 'snpca_id'
defense_configuration = deepcopy(all_defense_configurations[defense_type])
defense_configuration['type'] = defense_type

for t in range(my_data.num_classes):
    my_defense_configuration = deepcopy(defense_configuration)
    my_defense_configuration['target_class'] = t
    print(f'\n\nINFO: The target class is {my_defense_configuration['target_class']}, which corresponds to {my_data.get_class_names()[my_defense_configuration['target_class']]}.')
    _defense = NPCA_Custom(global_model.data, torch_model=global_model, defense_configuration=my_defense_configuration, verbose=False)
    _defense.defend()
    _defense.evaluate(poisoned_data)

