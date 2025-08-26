"""
Spectral Signature (SS) defense:
X0: clean features from TC
X1: attack features
Author: Zhen Xiang
Date: 6/15/2020
"""

# from __future__ import absolute_import
# from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as PCA_official


from utils_.pca import PCA_of_SKLEARN as PCA
from _0_general_ML.model_utils.torch_model import Torch_Model



# if not os.path.isdir('features'):
#     print('Extract features first!')
#     sys.exit(0)


class Spectral_Signatures_Official:
    
    def __init__(self, torch_model: Torch_Model):
        
        self.configuration = {
            'num_classes': torch_model.data.num_classes,
            'eps': 0.25,
            
        }
        
        self.torch_model = torch_model
        # self.threshold = self.configuration['threshold']
        
        self.decomp_all = {}
        self.means_for_ss = {}
        self.thresholds = {}
        
        pass
    
    
    def prepare_defense(
        self, 
        all_features_: np.ndarray, target_class: int,
        # clean_indices, poison_indices,
        poison_ratio: float=0.1,
        verbose: bool=False, 
        **kwargs
    ):
        
        # PCA
        # all_features_ = all_features_ - np.mean(all_features_, axis=0)
        decomp = PCA(all_features_, n_components=2, whiten=True)
        all_features_transformed = decomp.transform(all_features_)[:, 0]
        
        # SS defense metric
        ss_metric = np.square(all_features_transformed - np.mean(all_features_transformed))
        # rank = np.argsort(all_features_transformed)
        
        features_to_keep = round((1.5*poison_ratio)*len(ss_metric))
        
        self.thresholds[target_class] = np.sort(ss_metric)[-features_to_keep]
        self.decomp_all[target_class] = decomp
        self.means_for_ss[target_class] = np.mean(all_features_transformed)
        
        # # Visualize the internal layer activations projected onto the first two principal component
        # plt.hist(all_features_transformed[clean_indices], alpha=0.5, bins=100, label='clean')
        # plt.hist(all_features_transformed[poison_indices], alpha=0.5, bins=100, label='backdoor')
        # plt.axis('off')
        # plt.legend()
        # plt.show()
        # # plt.savefig('2D_separation.png')
        
        return
    
    
    def compute_ss_metric(self, features_, target_classes):
        all_features_transformed = [np.square(self.decomp_all[target_class].transform(features_[t:t+1])[:,0]-self.means_for_ss[target_class]) for t, target_class in enumerate(target_classes)]
        # all_features_transformed = [np.square(feature - self.means_for_ss[target_class]) for 
        return np.array(all_features_transformed).reshape(-1)
    def detect_with_ss_metric(self, features_, target_classes):
        all_features_transformed = self.compute_ss_metric(features_, target_classes)
        return np.array([all_features_transformed[t]>self.thresholds[target_class] for t, target_class in enumerate(target_classes)])
        
        
    def cleaning(
        self, 
        features_: np.ndarray, target_class: int, 
        clean_indices: list[int], poison_indices: list[int], 
        verbose: bool=False, 
        **kwargs
    ):
        
        # features_to_keep = round((1.5*poison_ratio)*len(all_features_transformed))
        # bad_indices = rank[-features_to_keep:]
        all_features_transformed = self.compute_ss_metric(features_, [target_class]*len(features_))
        bad_indices = self.detect_with_ss_metric(features_, [target_class]*len(features_))
        bad_indices = np.where(bad_indices)[0]
        
        if verbose:
            # Visualize the internal layer activations projected onto the first two principal component
            plt.hist(all_features_transformed[clean_indices], alpha=0.5, bins=100, label='clean')
            plt.hist(all_features_transformed[poison_indices], alpha=0.5, bins=100, label='backdoor')
            plt.axis('off')
            plt.legend()
            plt.show()
            # plt.savefig('2D_separation.png')
        
        # ----------------------------------------
        # will come back to this later
        # ----------------------------------------
        tp = 0
        for k in poison_indices:
            if k in bad_indices:
                tp += 1
        fp = 0
        for k in clean_indices:
            if k in bad_indices:
                fp += 1
        
        tp /= len(poison_indices)
        fp /= len(clean_indices)

        print(f'TPR: {tp}; FPR: {fp}')
        
        return
    
    
    def prepare_defense_old(
        self, 
        all_features_, 
        clean_indices, poison_indices, 
        poison_ratio: float=0.1,
        verbose: bool=False, 
        **kwargs
    ):
        
        # PCA
        # all_features_ = all_features_ - np.mean(all_features_, axis=0)
        decomp = PCA(all_features_, n_components=2, whiten=True)
        all_features_transformed = decomp.transform(all_features_)[:, 0]
        
        # SS defense
        ss_metric = np.square(all_features_transformed- np.mean(all_features_transformed))
        rank = np.argsort(ss_metric)
        
        features_to_keep = round((1.5*poison_ratio)*len(all_features_transformed))
        bad_indices = rank[-features_to_keep:]
        # self.threshold = all_features_transformed[features_to_keep]
        
        # ----------------------------------------
        # will come back to this later
        # ----------------------------------------
        if verbose:
            # Visualize the internal layer activations projected onto the first two principal component
            plt.hist(ss_metric[clean_indices], alpha=0.5, bins=100, label='clean')
            plt.hist(ss_metric[poison_indices], alpha=0.5, bins=100, label='backdoor')
            plt.axis('off')
            plt.legend()
            plt.show()
            # plt.savefig('2D_separation.png')
        
        tp = 0
        for k in poison_indices:
            if k in bad_indices:
                tp += 1
        fp = 0
        for k in clean_indices:
            if k in bad_indices:
                fp += 1
        
        tp /= len(poison_indices)
        fp /= len(clean_indices)

        print(f'TPR: {tp}; FPR: {fp}')
        
        
        return
        
        
    