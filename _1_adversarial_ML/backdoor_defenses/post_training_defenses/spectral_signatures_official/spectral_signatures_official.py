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
from sklearn.decomposition import PCA


from _0_general_ML.model_utils.torch_model import Torch_Model



# if not os.path.isdir('features'):
#     print('Extract features first!')
#     sys.exit(0)


class Spectral_Signatures_Official:
    
    def __init__(self, torch_model: Torch_Model):
        
        self.configuration = {
            'num_classes': torch_model.data.num_classes,
            'n_components': 2,
            'eps': 0.25,
            
        }
        
        self.torch_model = torch_model
        # self.threshold = self.configuration['threshold']
        
        pass
    
    
    def prepare_defense(
        self, 
        all_features_, 
        clean_indices, poison_indices, 
        poison_ratio: float=0.1,
        verbose: bool=False, 
        **kwargs
    ):
        
        # PCA
        all_features_ = all_features_ - np.mean(all_features_, axis=0)
        decomp = PCA(n_components=2, whiten=True)
        decomp.fit(all_features_)
        all_features_transformed = decomp.transform(all_features_)

        if verbose:
            # Visualize the internal layer activations projected onto the first two principal component
            plt.scatter(all_features_transformed[clean_indices, 0], all_features_transformed[clean_indices, 1], alpha=0.5, marker='o', label='clean')
            plt.scatter(all_features_transformed[poison_indices, 0], all_features_transformed[poison_indices, 1], alpha=0.5, marker='s', label='backdoor')
            plt.axis('off')
            plt.legend()
            plt.show()
            # plt.savefig('2D_separation.png')
        
        # SS defense
        all_features_transformed = np.square(all_features_transformed[:, 0] - np.mean(all_features_transformed[:, 0]))
        rank = np.argsort(all_features_transformed)
        
        features_to_keep = round((1.5*poison_ratio)*len(all_features_transformed))
        bad_indices = rank[-features_to_keep:]
        self.threshold = all_features_transformed[features_to_keep]
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
    
    