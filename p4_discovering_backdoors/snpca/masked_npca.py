import numpy as np

import torch
import torchvision
import copy

import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering, KMeans, HDBSCAN
from sklearn.metrics import f1_score, accuracy_score


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

# from .npca_paper import NPCA_Paper
from .npca_custom_with_masking import NPCA_Custom_with_Masking

from utils_.torch_utils import get_outputs, get_data_samples_from_loader, prepare_dataloader_from_numpy
from utils_.general_utils import normalize, exponential_normalize

from ..attacks.input_minimalist import Input_Minimalist
from ..attacks.input_minimalist_patch import Patch_Input_Minimalist



torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def input_minimalist_functions(
    minimalist_type: str, model: Torch_Model, minimalist_configuration: dict
) -> Input_Minimalist:
    
    all_input_minimalists = {
        'pixel_based': Input_Minimalist,
        'patch_based': Patch_Input_Minimalist
    }
    
    return all_input_minimalists[minimalist_type](model, minimalist_configuration)


class Masked_NPCA(NPCA_Custom_with_Masking):
    
    def __init__(
        self, 
        in_data: Torch_Dataset, 
        torch_model: Torch_Model, 
        audit_configuration: dict={},
        **kwargs
    ):
        
        default_audit_configuration = {
            'loss': 'crossentropy',
            'alpha': 0.5,
            'mask_ratio': 0.15,
            'patch_size': None,
            'n_clusters': 2,
            'iterations': 100,
            
        }
        for key in default_audit_configuration.keys():
            if key not in audit_configuration.keys():
                audit_configuration[key] = default_audit_configuration[key]
        
        super().__init__(in_data, torch_model, audit_configuration=audit_configuration)
        
        # preparing some useful functions
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.hdb = HDBSCAN(metric='l2', algorithm='auto', min_cluster_size=1, min_samples=1, allow_single_cluster=True)
        self.spectral = SpectralClustering(n_clusters=self.configuration['n_clusters'], affinity='precomputed')
        self.kmeans = KMeans(n_clusters=self.configuration['n_clusters'], n_init='auto')
        self.clusterer = self.spectral
        
        self.minimalist_type = 'patch_based' if self.configuration['patch_size'] is not None else 'pixel_based'
        self.data_loader_original = self.data_loader
        self.prepare_masked_data()
        
        return
    
    
    def prepare_masked_data(self):
        
        _x, y = get_data_samples_from_loader(self.data_loader, return_numpy=True)
        
        # min_attack = all_input_minimalists[self.minimalist_type](self.model, self.configuration)
        min_attack = input_minimalist_functions(self.minimalist_type, self.model, self.configuration)
        self.input_masks = min_attack.attack(_x, self.target_class*np.ones_like(y), iterations=self.configuration['iterations'])
        
        masked_data_loader = prepare_dataloader_from_numpy((min_attack.np_sigmoid(self.input_masks)*_x).copy(), y, batch_size=self.data_loader.batch_size)
        self.renew_configuration(data_loader=masked_data_loader)
        self.prepare_npca_things_original()
        
        return
    
    
    def sample_from_dataloader(self, return_numpy: bool=True, use_original_dataloader: bool=False, **kwargs):
        if use_original_dataloader:
            inputs, y = get_data_samples_from_loader(self.data_loader_original, return_numpy=return_numpy, **kwargs)
        else:
            inputs, y = get_data_samples_from_loader(self.data_loader, return_numpy=return_numpy, **kwargs)
        return inputs, y
    
    
    def analyze_the_spuriousity_of_the_feature(self, _mnpca: NPCA_Custom_with_Masking, top_n: int=1, max_feature: int=None):
        
        def positive_std(component_values: np.ndarray, feature_number: int=0):
            _values = component_values[:, feature_number]
            return np.std(_values[_values>0], axis=0)

        def check_all_present_features(_mnpca: NPCA_Custom_with_Masking, indices_=None):
            
            indices_ = np.arange(len(_mnpca.ac_.detach().cpu().numpy())) if indices_ is None else indices_
            component_values_ = _mnpca._pca.transform(_mnpca.ac_.detach().cpu().numpy()[indices_])
            component_values = np.clip(component_values_, 0, np.max(component_values_))
            
            return np.mean(np.sign(component_values), axis=0)
        
        features = check_all_present_features(_mnpca)
        metric = exponential_normalize(features)
        
        # max_feature = len(metric) if max_feature is None else max_feature
        max_feature = np.where(normalize(_mnpca._pca.variances)>0.05)[0][-1]
        metric = metric[:max_feature]
        
        feature_scores = np.argsort(metric)
        best_features = feature_scores[::-1][:top_n].tolist() # list(np.where(metric==np.max(metric))[0][:top_n])
        worst_features = feature_scores[:top_n].tolist() # list(np.where(metric==np.min(metric))[0][:top_n])
        
        return metric, best_features, worst_features
    
    
    def maximizing_train_points(
        self, pca_dim, k=5, metric_name: str='alpha_conf',
        device: str=torch_device, return_indices: bool=False
    ):
        
        objective = self.alpha_metric(pca_dim, device=device, metric_name=metric_name)
        sorted_idcs = torch.argsort(objective, descending=True)
        
        max_idcs = sorted_idcs[:k]
        max_images = []
        for idx in max_idcs:
            max_images.append(self.data_loader_original.dataset[idx][0])
        if return_indices:
            return max_images, max_idcs
        
        return max_images
    
    
    def metric_difference_from_original_features(self, clean_indices, poison_indices):
        
        metric = np.square(self._pca.reconstruct(self.ac_original.numpy()) - self.ac_.numpy())
        metric = np.mean(metric, axis=1)
        self.plot_a_metric(metric, clean_indices, poison_indices)
        
        return
    
    
    def metric_generated_mask(self, clean_indices, poison_indices):
        
        metric = np.mean(self.input_masks, axis=(1,2,3))
        self.plot_a_metric(metric, clean_indices, poison_indices)
        
        return
    
    
    