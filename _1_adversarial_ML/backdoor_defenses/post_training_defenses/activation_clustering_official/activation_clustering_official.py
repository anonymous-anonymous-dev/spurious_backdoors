# from __future__ import absolute_import
# from __future__ import print_function

# import os
# import sys
# import argparse

import torch
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score


from _0_general_ML.model_utils.torch_model import Torch_Model

from utils_.pca import PCA_of_SKLEARN as PCA
from utils_.torch_utils import get_data_samples_from_loader, get_outputs



class Activation_Clustering_Official:
    """
    Code taken and modified from: https://github.com/zhenxianglance/RE-paper/blob/main/AC_defense.py
    """
    
    def __init__(self, torch_model: Torch_Model):
        
        self.configuration = {
            'num_classes': torch_model.data.num_classes,
            'n_components': 2,
            # 'threshold': 0.4055,
            'threshold': 0.325 # for cifar10
        }
        
        self.torch_model = torch_model
        self.threshold = self.configuration['threshold']
        
        self.score_max = 0
        self.all_scores = np.zeros((self.configuration['num_classes']))
        self.kmeans_all = {}
        self.decomp_all = {}
        
        return
    
    
    def __activation_clustering_defense_custom(self, all_features_, target_class: int=0, verbose: bool=False, **kwargs):
        
        # AC detection
        score_max = 0
        all_scores = np.zeros((self.configuration['num_classes']))
        for i, class_ in enumerate(range(self.configuration['num_classes'])):
            print(f'\rAnalyzing class {class_}, which is {i+1}/{self.configuration['num_classes']}.', end='')
            
            all_features_ = all_features_ - np.mean(all_features_, axis=0)
            decomp = PCA(n_components=self.configuration['n_components'], whiten=True,)
            decomp.fit(all_features_)
            
            all_features_transformed = decomp.transform(all_features_)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(all_features_transformed)
            score = silhouette_score(all_features_transformed, kmeans.labels_)
            
            all_scores[class_] = score
            
            if score > score_max:
                self.most_vulnerable_class = class_
                score_max = score
                self.decomp_t = decomp
                self.kmeans_t = kmeans
        
        print()
        
        vulnerable_classes = np.where(all_scores>self.threshold)[0].reshape(-1)
        self.most_vulnerable_classes = np.where(all_scores==np.max(all_scores))[0].reshape(-1)
        if verbose:
            if (len(vulnerable_classes)>0):
                print(f'Attack detected ! The most vulnerable class is: {np.argmax(all_scores)} with score {np.max(all_scores):.5f}.')
                print(f'Other vulnerable classes are: {', '.join([f'\n{class_}, \t{all_scores[class_]:.5f}' for class_ in vulnerable_classes])}\n.')
            else:
                print('No backdoor class found.')
            
        
        return all_scores
    
    
    def activation_clustering_defense_custom(self, all_features_, target_class: int=0, save_stats: bool=False, verbose: bool=False, **kwargs):
        
        # AC detection
        if verbose:
            print(f'\rAnalyzing class {target_class}.', end='')
        
        decomp = PCA(all_features_, n_components=self.configuration['n_components'], whiten=True,)
        all_features_transformed = decomp.transform(all_features_)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(all_features_transformed)
        score = silhouette_score(all_features_transformed, kmeans.labels_)
        
        self.all_scores[target_class] = score
        self.kmeans_all[target_class] = {'algo': kmeans}
        self.decomp_all[target_class] = decomp
        
        if score > self.score_max:
            self.most_vulnerable_class = target_class
            self.score_max = score
        
        return
    
    
    def cleaning_model(self, clean_indices, poison_indices):
        
        kmeans_t = self.kmeans_all[self.most_vulnerable_class]['algo']
        
        # Training set cleansing
        TP_count = 0
        FP_count = 0
        # Remove the component with smaller mass
        count_10 = np.mean(kmeans_t.labels_[clean_indices])             # clean target_class samples labeled to cluster 1
        count_00 = 1 - count_10                                                     # clean target_class samples labeled to cluster 0
        count_11 = np.mean(kmeans_t.labels_[poison_indices])             # backdoor samples labeled to cluster 1
        count_01 = 1 - count_11                                                     # backdoor samples labeled to cluster 0
        if np.mean(kmeans_t.labels_[clean_indices]) > 0.5:
            TP_count = count_01
            FP_count = count_00
        else:
            TP_count = count_11
            FP_count = count_10
            
        TPR = TP_count # / len(backdoor_features)
        FPR = FP_count # / len(clean_features)
        
        print(f'TPR: {TPR:.4f}; FPR: {FPR:.4f}.')
        
        return TPR, FPR

