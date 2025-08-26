import torch
import numpy as np
from copy import deepcopy


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
    
    
class Adversarially_Smoothed_NPCA:
    
    def __init__(
        self,
        model: Torch_Model,
        defense_configuration: dict={}
    ):
        
        self.torch_model = model
        
        # new_model_ = Torch_Model_Save_Best(self.torch_model.data, model_configuration=self.torch_model.model_configuration)
        # new_model_.model.load_state_dict(self.torch_model.model.state_dict())
        # global_model = Torch_Model_Save_Best(poisoned_data, my_model_configuration, path=helper.save_path)
        # global_model.unfreeze_last_n_layers(n=None)
        # global_model.model.load_state_dict(new_model_.model.state_dict())
        self.feature_model = Dependable_Feature_Activations(self.torch_model, layer_numbers=[-1])
        
        self.reset_configurations(defense_configuration)
        
        return
    
    
    def reset_configurations(
        self, defense_configuration: dict={}
    ):
        
        self.defense_configuration = {
            'num_samples': 100,
            'iterations': 50,
            'perturbation_delta': 0.5,
            'layer_numbers': [-1],
            'threshold': 0.01, 'threshold_multiplier': 1.15,
            'target_class': 0,
            'n_components': 2
        }
        for key in defense_configuration.keys():
            self.defense_configuration[key] = defense_configuration[key]
            
        self.batch_size = self.torch_model.model_configuration['batch_size']
        
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
    
    
    def prepare_estimates_using_id_features(self, adversarial_analysis: bool=True):
        
        # pn_test_dl = torch.utils.data.DataLoader(self.torch_model.data.poisoned_test, batch_size=self.batch_size)
        # pn_x, pn_y = get_data_samples_from_loader(pn_test_dl, return_numpy=True)
        
        train_dl = torch.utils.data.DataLoader(self.torch_model.data.train, batch_size=self.batch_size)
        x, y = get_data_samples_from_loader(train_dl, return_numpy=True)
        
        condition = (y==self.defense_configuration['target_class'])
        # cl_test_dl = prepare_dataloader_from_numpy(x[condition], y[condition], batch_size=self.batch_size)
        # fm = Feature_Activations(global_model.model, global_model.get_children(global_model.model)[-1], target_class_=self.defense_configuration['target_class'])
        ac_, acp_ = self.get_penultimate_features(x[condition], y[condition], feature_model_=self.feature_model)
        
        # Learn a pca on clean samples only
        rf_use = acp_ if adversarial_analysis else ac_
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
    
    
    def prepare_estimates_using_ood_features(self, adversarial_analysis: bool=True):
        
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
        self.ood_model.train(epochs=5)
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
        ood_condition = (ood_y!=self.defense_configuration['target_class'])
        # ood_train_dl = prepare_dataloader_from_numpy(ood_x[ood_condition], ood_y[ood_condition], batch_size=self.batch_size)
        rf_, rfp_ = self.get_penultimate_features(ood_x[ood_condition], ood_y[ood_condition], feature_model_=self.feature_model_ood)
        rf_use = rfp_ if adversarial_analysis else rf_
        
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
    
    
    def __get_penultimate_features(self, dl, feature_model_: Dependable_Feature_Activations=None):
        
        feature_model_ = self.feature_model if feature_model_ is None else feature_model_
        
        _new_model = Torch_Model_Save_Best(self.ood_data, {})
        _new_model.model = deepcopy(feature_model_.model)
        
        x_, y_ = get_data_samples_from_loader(dl, return_numpy=True)
        
        attacker = FGSM(_new_model)
        xp = x_.copy()
        
        xps = []
        for i in range(1):
            noise = np.random.normal(0, 0.1*np.std(xp), size=xp.shape).astype(np.float32)
            # xp = xp + noise
            # xp = attacker.attack(xp, self.defense_configuration['target_class']*np.ones_like(y_), epsilon=0.3, iterations=50, targeted=False, verbose=False)
            xpn = attacker.attack(xp, self.defense_configuration['target_class']*np.ones_like(y_), epsilon=0.5, iterations=10, targeted=True, verbose=False)
            xps.append(xpn)
        xp = np.mean(xps, axis=0)
        p_dl = prepare_dataloader_from_numpy(xp, y_, batch_size=dl.batch_size)
        
        ac_ = get_outputs(feature_model_, dl, return_numpy=True)
        p_ac_ = get_outputs(feature_model_, p_dl, return_numpy=True)
        
        return ac_, p_ac_
    
    
    def __score(self, pca_: PCA_SKLEARN_MEDIAN, ac_, ref_):
        return np.mean((pca_.transform(ac_) - np.median(pca_.transform(ref_), axis=0, keepdims=True))**2, axis=1)
    
    
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
    
    
    def forward(self, x: np.ndarray, y: np.ndarray):
        # dl = prepare_dataloader_from_numpy(x, y, batch_size=self.batch_size)
        _, scores_, _ = self.analyze(x, y)
        return scores_
    
    