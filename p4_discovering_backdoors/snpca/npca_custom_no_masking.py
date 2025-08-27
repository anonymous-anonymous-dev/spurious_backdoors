import numpy as np
import torch
import copy
from termcolor import colored
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.fgsm import FGSM
from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor

from .npca_paper import NPCA_Paper
from .analyzer import PCA_Analyzer

from utils_.pca import PCA_Loss, PCA_of_SKLEARN, PCA_of_NPCA, PCA_SKLEARN_MEDIAN, Sparse_PCA_of_SKLEARN
from utils_.torch_utils import get_outputs, get_data_samples_from_loader, evaluate_on_numpy_arrays, prepare_dataloader_from_numpy
from utils_.general_utils import normalize, exponential_normalize, np_sigmoid

from ..attacks.input_minimalist import Input_Minimalist
from ..attacks.input_minimalist_patch import Patch_Input_Minimalist

from ..model_utils.quantizer import Quantization



torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def input_minimalist_functions(
    minimalist_type: str, model: Torch_Model, minimalist_configuration: dict
) -> Input_Minimalist:
    
    all_input_minimalists = {
        'pixel_based': Input_Minimalist,
        'patch_based': Patch_Input_Minimalist
    }
    
    return all_input_minimalists[minimalist_type](model, minimalist_configuration)


class NPCA_Custom(NPCA_Paper):
    
    def __init__(
        self, 
        data: Torch_Dataset=None, 
        torch_model: Torch_Model=None, 
        defense_configuration: dict={},
        verbose: bool=True,
        **kwargs
    ):
        
        self.defense_name = 'SNPCA'
        
        default_defense_configuration = {
            'subset_population': None,
            'repititions': 1,
            'sub_pca_components': 1,
            'number_of_best_and_worst_features': 3,
            'number_of_maximizing_samples_for_evaluating_feature_score': 10,
            'processing_type': '',
            'mitigation': 'AF', # we are not using this anymore
            'adversarial_epsilon': 0.5,
            
            'masking_configuration': {
                'alpha': 0.3,
                'mask_ratio': 0.7,
                'patch_size': 0.2,
                'iterations': 100,
            }
        }
        for key in default_defense_configuration.keys():
            if key not in defense_configuration.keys():
                defense_configuration[key] = default_defense_configuration[key]
        
        self.clusterer = KMeans(n_clusters=2, n_init='auto', init='random')
        
        super().__init__(data=data, torch_model=torch_model, defense_configuration=defense_configuration, verbose=verbose)
        self.prepare_npca_things_original()
        
        return
    
    
    def prepare_analyzer(self, *args, **kwargs) -> PCA_Analyzer:
        _xq, y = self.sample_from_dataloader(return_numpy=True, use_original_dataloader=True)
        transformed_loader = prepare_dataloader_from_numpy(_xq, y, batch_size=self.data_loader.batch_size)
        return PCA_Analyzer(
            *args, 
            # threshold=0,
            epsilon=self.configuration['adversarial_epsilon'],
            reference_loader=transformed_loader, 
            sample_subset=self.configuration['subset_population'], iterations=self.configuration['repititions'], 
            **kwargs
        )
    
    
    def defend(self, *args, adversarial_subset_ratio: float=0.1, use_original_dataloader: bool=True, show_plots: int=0, show_metrics=False, **kwargs):
        
        def extract_worst_features_from_sample_scores(sample_scores):
            # set variables from the dictionary
            number_of_best_and_worst_features=self.configuration['number_of_best_and_worst_features']
            num_of_maximizing_samples = 10 #self.configuration['number_of_maximizing_samples_for_evaluating_feature_score']
            # extract features
            act_pca = self._pca.transform(self.ac_.detach().cpu().numpy())
            # act_pca = np.array([normalize(k) for k in act_pca])
            pca_scores = []
            for pca_dim in range(act_pca.shape[1]):
                best_pca_indices = np.argsort(act_pca[:, pca_dim])[::-1][:num_of_maximizing_samples]
                pca_score_ = np.mean(-sample_scores[best_pca_indices]*normalize(act_pca[best_pca_indices, pca_dim]))
                # pca_score_ = np.mean(-_mnpca.new_scores[best_pca_indices])
                pca_scores.append(pca_score_)
            feature_scores = np.array(pca_scores)
            sorted_features = np.argsort(feature_scores)
            return copy.deepcopy(sorted_features[:number_of_best_and_worst_features]), copy.deepcopy(sorted_features[::-1][:number_of_best_and_worst_features]), feature_scores
        
        
        self.print_out(self.configuration)
        # state_before = self.key_flatten_client_state_np(self.model.model.state_dict())
        
        # ======================
        # Step 1: Quantifying the spuriousity of a sample
        # ======================
        x, y = self.sample_from_dataloader(return_numpy=True, use_original_dataloader=use_original_dataloader)
        self.analyzer = self.prepare_analyzer(self.model, self._model, x, y, self.ac_original.detach().cpu().numpy(), mode='one', verbose=self.verbose)
        self.final_model = self.model
        # ----------------------
        
        # ======================
        # Step 2: Quantifying the spuriousity of a feature
        # ======================
        self.scores = self.analyzer.scores_.copy()
        self.bad_features, self.good_features, self.feature_scores = extract_worst_features_from_sample_scores(self.scores)
        assert max(self.bad_features) <= min(self._pca.components.shape)
        # ----------------------
        
        # ======================
        # Step None: Computing metrics just for fun
        # ======================
        scores_2_bad = np.mean([self.alpha_metric(bad_feature).detach().cpu().numpy() for bad_feature in self.bad_features[:3]], axis=0)
        scores_2_good = np.mean([self.alpha_metric(good_feature).detach().cpu().numpy() for good_feature in self.good_features[:3]], axis=0)
        self.scores_2 = scores_2_bad - scores_2_good
        # ----------------------
        
        self.show_features_and_plots(use_original_dataloader=True, show_plots=show_plots, show_metrics=show_metrics)
        
        return
        
    
    def show_features_and_plots(self, use_original_dataloader: bool=False, show_plots: int=0, show_metrics: bool=True):
        
        self.print_out(
            '\nScore 1 stats: ', 
            colored(f'{np.mean(self.scores[self.clean_indices]):.3f}, {np.median(self.scores[self.clean_indices]):.3f}', 'green'), '\t|\t', 
            colored(f'{np.mean(self.scores[self.poison_indices]):.3f}, {np.median(self.scores[self.poison_indices]):.3f}', 'red'),
            # verbose=True
        )
        self.print_out(
            f'Score 2 stats: ', 
            colored(f'{np.mean(self.scores_2[self.clean_indices]):.3f}, {np.median(self.scores_2[self.clean_indices]):.3f}', 'green'), '\t|\t', 
            colored(f'{np.mean(self.scores_2[self.poison_indices]):.3f}, {np.median(self.scores_2[self.poison_indices]):.3f}', 'red'), 
            # verbose=True
        )
        
        if show_plots>0:
            self.plot_components(show_features=list(self.good_features[:show_plots])+list(self.bad_features[:show_plots]), num_images_for_each_feature=10, use_original_dataloader=use_original_dataloader, save_fig=False);
            plt.show()
        if show_metrics:
            self.plot_metrics();
            plt.show()
        
        return
    
    
    def evaluate(self, data_in: Simple_Backdoor, *args, **kwargs):
        
        if self.analyzer.poisoning_score > self.analyzer.threshold:
            clean_dataloader = torch.utils.data.DataLoader(data_in.test, batch_size=self.batch_size, shuffle=False)
            poisoned_dataloader = torch.utils.data.DataLoader(data_in.poisoned_test, batch_size=self.batch_size, shuffle=False)
            
            xc, yc = get_data_samples_from_loader(clean_dataloader, return_numpy=True)
            xp, yp = get_data_samples_from_loader(poisoned_dataloader, return_numpy=True)
            
            yc_out = get_outputs(self.model.model, clean_dataloader, return_numpy=True)
            yp_out = get_outputs(self.model.model, poisoned_dataloader, return_numpy=True)
            
            yc_out = self.analyzer.forward(xc, yc_out)
            yp_out = self.analyzer.forward(xp, yp_out)
            
            yc_class = np.argmax(yc_out, axis=1)
            yp_class = np.argmax(yp_out, axis=1)
            poison_eval_indices = (yc_class==yc) & (yc!=self.target_class)
            
            acc_c = np.mean(yc_class == yc)
            acc_p = np.mean(yp_class[poison_eval_indices] == yp[poison_eval_indices])
            
            loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
            loss_c = loss_function(torch.tensor(yc_out), torch.tensor(yc))
            loss_p = loss_function(torch.tensor(yp_out[poison_eval_indices]), torch.tensor(yp[poison_eval_indices]))
                
            print(colored(f'Loss clean {loss_c:.3f}, Accuracy clean {acc_c:.3f}, Loss poisoned {loss_p:.3f}, Accuracy poisoned {acc_p:.3f}', 'green'))
        
            return (loss_c, acc_c), (loss_p, acc_p)

        return super().evaluate(data_in)
    
    
    def defend_mr(self, *args, **kwargs): return self.not_implemented_error()
    def prepare_npca_things_original(self):
        self.ac_original = self.ac_ #get_outputs(self._model, self.data_loader_original)
        self._pca_original = self._pca #self.pca_functions(self.ac_original.detach().cpu().numpy(), n_components=self.configuration['n_components'])
        return
    
    
    def plot_metrics(self, *args, **kwargs):
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 3))
        
        # if 'kaggle_imagenet' not in self.data.data_name:
        if (self.configuration['non_target_class'] is None) or (self.configuration['non_target_class'] == self.configuration['target_class']):
            axs[0].hist(self.scores[self.clean_indices], bins=100, alpha=0.5, color='green', label='clean')
            axs[0].hist(self.scores[self.poison_indices], bins=100, alpha=0.5, color='red', label='poison')
        else:
            axs[0].hist(self.scores, bins=100, color='gray', label='all')
        axs[0].legend()
        axs[0].set_xlabel('Difference from Reconstructed Samples')
        axs[0].set_ylabel('Number of Samples')
        
        if (self.configuration['non_target_class'] is None) or (self.configuration['non_target_class'] == self.configuration['target_class']):
            axs[1].hist(self.scores_2[self.clean_indices], bins=100, alpha=0.5, color='green', label='clean')
            axs[1].hist(self.scores_2[self.poison_indices], bins=100, alpha=0.5, color='red', label='poison')
        else:
            axs[1].hist(self.scores_2, bins=100, color='gray', label='all')
        axs[1].legend()
        axs[1].set_xlabel('Spurioustiy')
        axs[1].set_ylabel('Number of Samples')
        
        axs[2].plot(-self.feature_scores)
        axs[2].set_ylabel('Spuriousity')
        axs[2].set_xlabel('Features')
        
        axs[3].hist(-self.feature_scores, bins=100);
        axs[3].set_xlabel('Spuriousity of Features')
        axs[3].set_ylabel('Number of Features')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    
    def key_flatten_client_state_np(self, client_state_dict: dict):
        
        flattened_client_state = []
        for key in client_state_dict.keys():
            flattened_client_state += client_state_dict[key].clone().cpu().flatten().tolist()
        
        return np.array(flattened_client_state)
    

    def key_unflatten_client_state_np(self, flattened_client_state):
        
        client_state_dict_ = copy.deepcopy(self.original_model.model.state_dict())
            
        flattened_client_state_copy = torch.tensor(flattened_client_state.copy())
        unflattened_client_state = {}
        for key in client_state_dict_.keys():
            np_state_key = client_state_dict_[key].cpu().numpy()
            unflattened_client_state[key] = flattened_client_state_copy[:len(np_state_key.flatten())].reshape(np_state_key.shape)
            flattened_client_state_copy = flattened_client_state_copy[len(np_state_key.flatten()):]
        
        return unflattened_client_state
    
    
    