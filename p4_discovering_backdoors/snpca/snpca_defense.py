import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from termcolor import colored
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
# from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
# from _0_general_ML.model_utils.torch_model import Torch_Model

# from _1_adversarial_ML.adversarial_attacks.fgsm import FGSM
from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor
from _1_adversarial_ML.backdoor_defenses.post_training_defenses.backdoor_defense import Backdoor_Detection_Defense

# from .analyzer import Adversarially_Smoothed_NPCA

# from utils_.pca import PCA_Loss, PCA_of_SKLEARN, PCA_of_NPCA, PCA_SKLEARN_MEDIAN, Sparse_PCA_of_SKLEARN
# from utils_.torch_utils import get_outputs, get_data_samples_from_loader, evaluate_on_numpy_arrays, prepare_dataloader_from_numpy
# from utils_.general_utils import normalize, exponential_normalize, np_sigmoid

# from ..attacks.input_minimalist import Input_Minimalist
# from ..attacks.input_minimalist_patch import Patch_Input_Minimalist

# from ..model_utils.quantizer import Quantization

from utils_.general_utils import normalize
from utils_.torch_utils import prepare_dataloader_from_numpy, prepare_dataloader_from_tensor, get_outputs, get_data_samples_from_loader
from utils_.pca import PCA_of_SKLEARN, PCA_SKLEARN_MEDIAN

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset, Client_Torch_SubDataset
from _0_general_ML.data_utils.datasets import GTSRB, CIFAR10, CIFAR100

from _0_general_ML.model_utils.torch_model import Torch_Model
from _0_general_ML.model_utils.generalized_model_activations_wrapper import Dependable_Feature_Activations

from _1_adversarial_ML.adversarial_attacks.all_available_adversarial_attacks import FGSM

from ..model_utils.torch_model_save_best import Torch_Model_Save_Best



class ASNPCA_Stats:
    
    def __init__(self):
        self.pca_analyzers = []
        self.pc_maxs = []
        self.pc_mins = []
        self.pc_p_maxs = []
        self.pc_p_mins = []
        self.pc_medians = []
        self.pcs_s = []
        self.scores_s = []
        self.scores_maxs = []
        self.scores_mins = []
        return
    
    def update_stats(self, pca_analyzer: PCA_of_SKLEARN, activations: np.ndarray, p_activations: np.ndarray):
        pc = pca_analyzer.transform(activations).reshape(len(activations), -1)
        pc_p = pca_analyzer.transform(p_activations).reshape(len(p_activations), -1)
        pc_max = np.max(pc)
        pc_min = np.min(pc)
        pc_p_max = np.max(pc_p)
        pc_p_min = np.min(pc_p)
        pc = (pc-pc_min)/(pc_max-pc_min); #pc = pc - np.median(pc)
        pc_p =(pc_p-pc_p_min)/(pc_p_max-pc_p_min); #pc_p = pc_p - np.median(pc_p)
        
        pcs_ = np.append(pc, pc_p, axis=1)
        pc_median = np.median(pcs_, axis=0, keepdims=True)
        pcs_ = pcs_ - pc_median
        
        scores_ = np.mean(pcs_**2, axis=1)
        # scores_ += np.mean((p_activations-pca_analyzer.reconstruct(p_activations))**2, axis=1)
        scores_ += np.mean((activations-pca_analyzer.reconstruct(activations))**2, axis=1)
        scores_max = np.max(scores_)
        scores_min = np.min(scores_)
        scores_ = (scores_-scores_min)/(scores_max-scores_min)
        
        self.pca_analyzers.append(pca_analyzer)
        self.pc_maxs.append(pc_max)
        self.pc_mins.append(pc_min)
        self.pc_p_maxs.append(pc_p_max)
        self.pc_p_mins.append(pc_p_min)
        self.pc_medians.append(pc_median)
        self.scores_s.append(scores_)
        self.scores_maxs.append(scores_max)
        self.scores_mins.append(scores_min)
        self.pcs_s.append(pcs_)
        return
    
    
    def compute_pcs_and_score(self, activations: np.ndarray, p_activations: np.ndarray, i: int):
        pc = self.pca_analyzers[i].transform(activations).reshape(len(activations), -1)
        pc_p = self.pca_analyzers[i].transform(p_activations).reshape(len(p_activations), -1)

        pc = (pc-self.pc_mins[i])/(self.pc_maxs[i]-self.pc_mins[i]); #pc = pc - np.median(pc)
        pc_p =(pc_p-self.pc_p_mins[i])/(self.pc_p_maxs[i]-self.pc_p_mins[i]); #pc_p = pc_p - np.median(pc_p)
        pcs_ = np.append(pc, pc_p, axis=1)
        pcs_ = pcs_ - self.pc_medians[i]
        
        scores_ = np.mean(pcs_**2, axis=1)
        # scores_ += np.mean((p_activations-self.pca_analyzers[i].reconstruct(p_activations))**2, axis=1)
        scores_ += np.mean((activations-self.pca_analyzers[i].reconstruct(activations))**2, axis=1)
        scores_ = (scores_-self.scores_mins[i])/(self.scores_maxs[i]-self.scores_mins[i])
        
        return pcs_, scores_
    
    
class Adversarially_Smoothed_NPCA_Defense(Backdoor_Detection_Defense):
    
    def __init__(
        self,
        model: Torch_Model,
        defense_configuration: dict={}
    ):
        
        super().__init__(model, defense_configuration)
        
        return
    
    
    def configure_defense(
        self, *args, defense_configuration: dict={}, **kwargs
    ):
        
        self.defense_configuration = {
            'num_samples': 100,
            'iterations': 50,
            'perturbation_delta': 0.5,
            'layer_numbers': [-1],
            'threshold': 0.01, 'threshold_multiplier': 1.15,
            'target_class': 0,
            'n_components': 1,
            'epochs': 3,
            'adversarial_analysis': False
        }
        for key in defense_configuration.keys():
            self.defense_configuration[key] = defense_configuration[key]
            
        self.feature_model = Dependable_Feature_Activations(self.torch_model, layer_numbers=[-1])
        self.batch_size = self.torch_model.model_configuration['batch_size']
        self.target_class = self.defense_configuration['target_class']
        self.adversarial_analysis = self.defense_configuration['adversarial_analysis']
        
        self.pca_stats = ASNPCA_Stats()
        
        # =====================
        # Prepare OOD data
        # =====================
        if 'gtsrb' not in self.torch_model.model_configuration['dataset_name']:
            self.ood_data = GTSRB(preferred_size=self.torch_model.data.preferred_size, data_means=self.torch_model.data.data_means, data_stds=self.torch_model.data.data_stds)
        else:
            self.ood_data = CIFAR100(preferred_size=self.torch_model.data.preferred_size, data_means=self.torch_model.data.data_means, data_stds=self.torch_model.data.data_stds)
        indices_train = np.where(np.array(self.ood_data.train.targets).astype('int') < self.torch_model.data.num_classes)[0]
        self.ood_data.train = Client_SubDataset(self.ood_data.train, indices=indices_train)
        indices_test = np.where(np.array(self.ood_data.test.targets).astype('int') < self.torch_model.data.num_classes)[0]
        self.ood_data.test = Client_SubDataset(self.ood_data.test, indices=indices_test)
        
        # check if the dataset is available or not
        if self.torch_model.data.train is None:
            self.data_mode = False
            self.prepare_estimates_using_ood_features()
        else:
            self.prepare_estimates_using_id_features()
        
        return
    
    
    def prepare_estimates_using_id_features(self):
        
        # pn_test_dl = torch.utils.data.DataLoader(self.torch_model.data.poisoned_test, batch_size=self.batch_size)
        # pn_x, pn_y = get_data_samples_from_loader(pn_test_dl, return_numpy=True)
        
        train_dl = torch.utils.data.DataLoader(self.torch_model.data.train, batch_size=self.batch_size)
        x, y = get_data_samples_from_loader(train_dl, return_numpy=True)
        
        condition = (y==self.defense_configuration['target_class'])
        # cl_test_dl = prepare_dataloader_from_numpy(x[condition], y[condition], batch_size=self.batch_size)
        # fm = Feature_Activations(global_model.model, global_model.get_children(global_model.model)[-1], target_class_=self.defense_configuration['target_class'])
        ac_, acp_ = self.get_penultimate_features(x[condition], y[condition], feature_model_=self.feature_model)
        
        # Learn a pca on clean samples only
        rf_use = acp_ if self.adversarial_analysis else ac_
        for k in range(self.defense_configuration['iterations']):
            pca = PCA_SKLEARN_MEDIAN(
                rf_use[np.random.choice(len(rf_use), self.defense_configuration['num_samples'], replace=True)], 
                n_components=self.defense_configuration['n_components'], normalization=True
            )
            self.pca_stats.update_stats(pca, ac_, acp_)
            
        self.scores_ = np.mean(self.pca_stats.scores_s, axis=0)
        self.score_threshold = np.sort(self.scores_)[-int(self.defense_configuration['threshold']*len(self.scores_))]
        self.rf_use = rf_use
        
        return
    
    
    def prepare_estimates_using_ood_features(self):
        
        # ======================
        # Prepare OOD Model
        # ======================
        self.ood_model = Torch_Model_Save_Best(self.ood_data, model_configuration=self.torch_model.model_configuration)
        self.ood_model.model.load_state_dict(self.torch_model.model.state_dict())
        # new_model.model.load_state_dict(self.torch_model.model.state_dict())
        # test_on_clean_and_poisoned_data(new_model, my_data, poisoned_data, helper); print()
        self.ood_model.freeze_last_n_layers(n=None)
        self.ood_model.unfreeze_last_n_layers(n=20)
        self.ood_model.freeze_last_n_layers(n=10)
        # self.ood_model.unfreeze_last_n_layers(n=10)
        # self.ood_model.freeze_last_n_layers(n=5)
        self.ood_model.train(epochs=self.defense_configuration['epochs'])
        # test_on_clean_and_poisoned_data(self.ood_model, my_data, poisoned_data, helper); print()
        # test_on_clean_and_poisoned_data(global_model, gtsrb_data, poisoned_data, helper); print()
        
        # =======================
        # Prepare OOD Features
        # =======================
        new_model_ = Torch_Model_Save_Best(self.ood_data, model_configuration=self.torch_model.model_configuration)
        new_model_.model.load_state_dict(self.ood_model.model.state_dict())
        self.ood_model = Torch_Model_Save_Best(self.ood_data, model_configuration=self.torch_model.model_configuration)
        self.ood_model.model.load_state_dict(new_model_.model.state_dict())
        self.feature_model_ood = Dependable_Feature_Activations(self.ood_model, layer_numbers=[-1])
        
        ood_train_dl = torch.utils.data.DataLoader(self.ood_data.train, batch_size=self.batch_size)
        ood_x, ood_y = get_data_samples_from_loader(ood_train_dl, return_numpy=True)
        ood_condition = (ood_y==self.defense_configuration['target_class'])
        # ood_train_dl = prepare_dataloader_from_numpy(ood_x[ood_condition], ood_y[ood_condition], batch_size=self.batch_size)
        rf_, rfp_ = self.get_penultimate_features(ood_x[ood_condition], ood_y[ood_condition], feature_model_=self.feature_model_ood)
        rf_use = rfp_ if self.adversarial_analysis else rf_
        
        # ========================
        # Train OOD NPCA and set the threshold
        # ========================
        for k in range(self.defense_configuration['iterations']):
            pca = PCA_SKLEARN_MEDIAN(
                rf_use[np.random.choice(len(rf_use), self.defense_configuration['num_samples'], replace=True)], 
                n_components=self.defense_configuration['n_components'], normalization=True
            )
            self.pca_stats.update_stats(pca, rf_, rfp_)
            
        self.scores_ = np.mean(self.pca_stats.scores_s, axis=0)
        # rft_ = self.score(self.pca, rf_use, rf_use)
        self.score_threshold = np.sort(self.scores_)[-int(self.defense_configuration['threshold']*len(self.scores_))]
        self.rf_use = rf_use
        
        return
    
    
    def get_penultimate_features(self, x, y, feature_model_: Dependable_Feature_Activations=None):
        
        feature_model_ = self.feature_model if feature_model_ is None else feature_model_
        
        _new_model = Torch_Model_Save_Best(self.ood_data, {})
        _new_model.model = deepcopy(feature_model_.model)
        attacker = FGSM(_new_model)
        
        # noise = np.random.normal(0, 0.2*np.std(x), size=x.shape).astype(np.float32)
        # x = x + noise
        xp = attacker.attack(x, self.defense_configuration['target_class']*np.ones_like(y), epsilon=0.5, targeted=True, verbose=False)
        
        # _mnpca._model.mode = 'only_activations'
        activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x, y, batch_size=self.batch_size), return_numpy=True)
        p_activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(xp, y, batch_size=self.batch_size), return_numpy=True)
        # _mnpca._model.mode = 'default'
            
        return activations, p_activations
    
    
    def analyze(self, x, y):
        
        activations, p_activations = self.get_penultimate_features(x, y)
        
        pcs_s = []; scores_s = []
        for i in range(self.defense_configuration['iterations']):
            pcs_, scores_ = self.pca_stats.compute_pcs_and_score(activations, p_activations, i)
            pcs_s.append(pcs_); scores_s.append(scores_)
            
        pcs_s = np.mean(pcs_s, axis=0)
        scores_s = np.mean(scores_s, axis=0)
        # metric_scores = np.append(pcs_s, scores_s.reshape(-1, 1), axis=1)
        # label = self.clusterer.predict(metric_scores)
    
        return pcs_s, scores_s, 0 #label if self.poisoning_score>self.threshold else self.good_label*np.ones_like(label)
    
    
    def forward(self, x: np.ndarray, y_out: np.ndarray):
        
        y = np.argmax(y_out, axis=1)
        pc, score, label = self.analyze(x, y)
        
        vulnerable_ind = (y==self.target_class)
        vulnerable_ind = vulnerable_ind & (score>self.score_threshold)
        
        y_random = np.random.normal(0, self.torch_model.data.num_classes, size=y_out.shape)
        y_random[y_random==self.target_class] += 1
        y_random = y_random % self.torch_model.data.num_classes
        y_out[vulnerable_ind] = y_random[vulnerable_ind]
        
        return y_out
    
    
    def evaluate(self, data_in: Simple_Backdoor, *args, **kwargs):
        
        self.model_is_poisoned = True
        
        if self.model_is_poisoned:
            clean_dataloader = torch.utils.data.DataLoader(data_in.test, batch_size=self.batch_size, shuffle=False)
            poisoned_dataloader = torch.utils.data.DataLoader(data_in.poisoned_test, batch_size=self.batch_size, shuffle=False)
            
            xc, yc = get_data_samples_from_loader(clean_dataloader, return_numpy=True)
            xp, yp = get_data_samples_from_loader(poisoned_dataloader, return_numpy=True)
            
            yc_out = get_outputs(self.torch_model.model, clean_dataloader, return_numpy=True)
            yp_out = get_outputs(self.torch_model.model, poisoned_dataloader, return_numpy=True)
            
            yc_out = self.forward(xc, yc_out)
            yp_out = self.forward(xp, yp_out)
            
            yc_class = np.argmax(yc_out, axis=1)
            yp_class = np.argmax(yp_out, axis=1)
            poison_eval_indices = (yc_class==yc) & (yc!=self.target_class)
            
            acc_c = np.mean(yc_class == yc)
            acc_p = np.mean(yp_class[poison_eval_indices] == yp[poison_eval_indices])
            
            loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
            loss_c = loss_function(torch.tensor(yc_out), torch.tensor(yc))
            loss_p = loss_function(torch.tensor(yp_out[poison_eval_indices]), torch.tensor(yp[poison_eval_indices]))
                
            print(f'Loss clean {loss_c:.3f}, Accuracy clean {acc_c:.3f}, Loss poisoned {loss_p:.3f}, Accuracy poisoned {acc_p:.3f}')
        
            return (loss_c, acc_c), (loss_p, acc_p)

        return super().evaluate(data_in)
    
    