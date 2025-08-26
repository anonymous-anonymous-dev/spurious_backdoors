import torch
import numpy as np
import math
from termcolor import colored

from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


from _0_general_ML.data_utils.torch_data import Torch_Data
from _0_general_ML.model_utils.torch_model import Torch_Model
from _1_adversarial_ML.adversarial_attacks.all_available_adversarial_attacks import FGSM
from .fgsm_torch import FGSM_Torch

from ..model_utils.feature_activations import Feature_Activations

from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy, get_outputs, prepare_dataloader_from_tensor
from utils_.pca import PCA_of_SKLEARN

from .analyzer import ASNPCA_Stats, PCA_Analyzer



class My_Clusterer_Original:
    
    def __init__(self):
        self.configured = False
        self.lr = LinearRegression()
        return
    
    
    def sigmoid(self, value: float):
        return 1/(1+np.exp(-20 * value))
    def get_ellipse(self):
        return Ellipse(xy=(self.center_x, self.center_y), width=self.width, height=self.height, angle=self.angle_degrees, edgecolor='orange', facecolor='none', linewidth=2)
    
    
    def configure(self, center_x, center_y, width, height, angle_degrees):
        
        self.configured = True
        
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.angle_degrees = angle_degrees
        
        return
    
    
    def fit(self, limited_pa: np.ndarray):
        
        xs = limited_pa[:, 0:1]
        ys = limited_pa[:, 1:2]
        multiplied_ = limited_pa
        
        # self.ref_pcs -= np.median(self.ref_pcs, axis=0, keepdims=True)
        self.lr.fit(xs, ys)
        coefficients = self.lr.coef_
        self.original_theta = np.arctan2(coefficients[0, 0], 1)
        self.theta = self.original_theta
        if self.theta<0:
            self.theta += 2*np.pi
        if self.theta>(np.pi/4) and self.theta<=(3*np.pi/4):
            self.theta -= np.pi/2
        elif self.theta>(3*np.pi/4) and self.theta<=(5*np.pi/4):
            self.theta -= np.pi
        elif self.theta>(5*np.pi/4) and self.theta<=(7*np.pi/4):
            self.theta -= 3*np.pi/2
        self.matrix_multiplier = np.array([
            np.array([np.cos(self.theta), -np.sin(self.theta)]),
            np.array([np.sin(self.theta), np.cos(self.theta)])
        ]).astype(limited_pa.dtype)
        # self.matrix_multiplier = self.matrix_multiplier.astype(np.float32)
        self.inverse_matrix_multiplier = np.array([
            np.array([np.cos(-self.theta), -np.sin(-self.theta)]),
            np.array([np.sin(-self.theta), np.cos(-self.theta)])
        ])
        
        new_multiplied = multiplied_ @ self.matrix_multiplier
        
        center_x = np.median(xs)  # X-coordinate of the ellipse center
        center_y = np.median(ys)  # Y-coordinate of the ellipse center
        angle_degrees = (self.theta*180/np.pi)  # Rotation angle in degrees, anti-clockwise
        width = 4 * np.std(new_multiplied[:,0])
        height = 4 * np.std(new_multiplied[:,1])
        # width = 4 * np.sqrt(np.median((new_multiplied[:,0]-np.mean(new_multiplied[:,0]))**2))
        # height = 4 * np.sqrt(np.median((new_multiplied[:,1]-np.mean(new_multiplied[:,1]))**2))
        
        self.configure(center_x, center_y, width, height, angle_degrees)
        
        self.center = np.array([self.center_x, self.center_y]).reshape(1,2)
        self.right_most_point = np.dot(np.array([width/2, 0]).reshape(1, 2), self.inverse_matrix_multiplier) + self.center
        self.left_most_point = np.dot(np.array([-width/2, 0]).reshape(1, 2), self.inverse_matrix_multiplier) + self.center
        if self.right_most_point[0,0]<self.left_most_point[0,0]:
            right_most_point = self.left_most_point.copy()
            self.left_most_point = self.right_most_point.copy()
            self.right_most_point = right_most_point.copy()
        self.ellipse_slope = (self.right_most_point[0,1]-self.left_most_point[0,1])/(self.right_most_point[0,0]-self.left_most_point[0,0])
        self.intercept = self.left_most_point[0,1] - self.ellipse_slope*self.left_most_point[0,0]
        
        return
    
    
    def calculate_the_radius_from_normalized_ellipse(self, point_x, point_y):
        """
        Checks if a point is inside an ellipse.
        
        Args:
            point_x, point_y (float): Coordinates of the point to check.
            
        Returns:
            bool: True if the point is inside or on the ellipse, False otherwise.
        """
        # 1. Translate the point
        translated_x = point_x - self.center_x
        translated_y = point_y - self.center_y
        
        # 2. Rotate the point (negative angle for counter-clockwise rotation)
        angle_rad = math.radians(-self.angle_degrees)
        rotated_x = translated_x * math.cos(angle_rad) - translated_y * math.sin(angle_rad)
        rotated_y = translated_x * math.sin(angle_rad) + translated_y * math.cos(angle_rad)
        
        # Calculate semi-axes
        semi_major_axis = self.width / 2
        semi_minor_axis = self.height / 2
        
        # 3. Check the ellipse equation
        value_1 = (rotated_x**2/semi_major_axis**2)
        value_2 = (rotated_y**2/semi_minor_axis**2)
        value = value_1 + value_2
        
        return value
    
    
    def predict(self, point_x, point_y):
        
        ys_at_this_point = self.ellipse_slope*point_x + self.lr.intercept_
        # condition_clean_0 = (point_x>=self.right_most_point[0,0]) & (point_y<=self.right_most_point[0,1])
        condition_clean_1 = (point_x<=self.left_most_point[0,0]) & (point_y<=self.left_most_point[0,1])
        condition_clean_2 = (point_x>=self.left_most_point[0,0]) & (point_x<=self.right_most_point[0,0]) & (ys_at_this_point>=point_y)
        
        radius_original = self.calculate_the_radius_from_normalized_ellipse(point_x, point_y)
        label = self.sigmoid(radius_original-1)
        
        ones = np.ones_like(label.reshape(-1,1))
        # label[condition_clean_0] = np.min(np.append(label.reshape(-1,1), 0.25*ones, axis=1), axis=1)[condition_clean_0]
        label[condition_clean_1] = np.min(np.append(label.reshape(-1,1), 0.1*ones, axis=1), axis=1)[condition_clean_1]
        label[condition_clean_2] = np.min(np.append(label.reshape(-1,1), 0.1*ones, axis=1), axis=1)[condition_clean_2]
        # label[condition_clean_0] = 0.3
        # label[condition_clean_1] = 0.1
        # label[condition_clean_2] = 0.1
        
        return label
    
    
class My_Clusterer:
    
    def __init__(self, iterations: int=5):
        self.configured = False
        self.iterations = iterations
        self.ellipses = [My_Clusterer_Original()] * self.iterations
        return
    
    def fit(self, limited_pa: np.ndarray):
        self.iterations = np.clip(len(limited_pa)//500, 1, 5)
        for i in range(self.iterations):
            _limited_pa = limited_pa[np.random.choice(len(limited_pa), size=min(500, len(limited_pa)), replace=False)]
            self.ellipses[i].fit(_limited_pa)
        self.configured = True
        return
    
    def predict(self, point_x, point_y):
        labels = []
        for i in range(self.iterations):
            labels.append(self.ellipses[i].predict(point_x, point_y))
        labels = np.array(labels)
        labels = np.mean(labels, axis=0)
        return labels
    
    
class PCA_Analyzer_Random(PCA_Analyzer):
    
    def __init__(
        self, 
        model: Torch_Model, feature_model: Feature_Activations, 
        x, y, ac_, 
        epsilon=0.5, 
        mode='one', sample_subset: int=None, iterations: int=100, std: np.float32=0.2,
        reference_loader: torch.utils.data.DataLoader=None,
        threshold: float=0.,
        verbose: bool=False
    ):
        
        self.correction = 0
        self.reference_loader = reference_loader
        self.clusterer_hdb = HDBSCAN(min_cluster_size=len(reference_loader.dataset), cluster_selection_epsilon=0.1)
        self.custom_clusterer_random = My_Clusterer()
        self.ref_metric_scores = None
        self.matrix_multiplier = None
        
        super().__init__(
            model, feature_model, x, y, ac_,
            epsilon=epsilon, 
            reference_loader=reference_loader,
            mode=mode, sample_subset=sample_subset, iterations=iterations, std=std,
            threshold=0.,
            verbose=verbose
        )
        
        return
    
    
    def prepare_clusterers(self):
        
        self.custom_clusterer = My_Clusterer()
        self.good_label = 0; self.bad_label = 1;
        self.poisoning_score = self.threshold + 1
        
        limited_x, limited_y = get_data_samples_from_loader(self.reference_loader, return_numpy=True)
        limited_pa = []
        for i in range(50):
            _limited_pa, _limited_scores, _limited_labels = self.analyze(limited_x, limited_y)
            limited_pa.append(_limited_pa)
        self.limited_pa_ = np.concatenate(limited_pa, axis=0)
        self.limited_pa = np.median(limited_pa, axis=0)
        
        self.custom_clusterer.fit(self.limited_pa_)
        
        return
    
    
    def prepare_analyzer_smooth(self, x, y):
        
        xr, yr = get_data_samples_from_loader(self.reference_loader, return_numpy=True)
        xr, yr = xr[yr==self.target_class], yr[yr==self.target_class]
        r_activations, rp_activations = self.get_penulatimate_features(xr, yr)
        self.pca_stats = ASNPCA_Stats(use_median=False, reference_activations=[r_activations, rp_activations])
        
        super().prepare_analyzer_smooth(x, y)
        
        # self.pcs_, self.scores_, _ = self.analyze(x, y)
        # metric_scores = np.append(self.pcs_[:, -2:], self.scores_.reshape(-1, 1), axis=1)
        # ref_metric_scores = np.append(ref_pcs[:, -2:], ref_scores.reshape(-1, 1), axis=1)
        # metric_scores = np.append(metric_scores, ref_metric_scores, axis=0)
        # self.clusterer.fit(metric_scores)
        # labels = self.clusterer.labels_
        # self.clusterer.cluster_centers_ = self.clusterer.cluster_centers_.astype(np.float32)
        
        # # label_values = [np.mean(self.scores_[labels==0]), np.mean(self.scores_[labels!=0])]
        # # self.good_label = np.argmin(label_values); self.bad_label = np.argmax(label_values)
        # self.good_label = np.median(self.clusterer.predict(ref_metric_scores))
        # self.bad_label = 1 - self.good_label
        
        # # strategy 2 for poisoning score
        # all_scores = np.append(self.scores_.reshape(-1, 1), ref_scores.reshape(-1, 1), axis=0)
        # good_l = (labels==self.good_label)
        # bad_l = (labels==self.bad_label)
        # max_l = np.mean(good_l)
        # self.poisoning_score = np.mean(all_scores[bad_l]) - np.mean(all_scores[good_l]) * max_l
        
        return
    
    
    def analyze(self, x, y):
        pc_, score_, labels = super().analyze(x, y, cluster_xy=False)
        
        # ==============
        # Strategy 1
        # ==============
        # labels = np.array([self.bad_label if pc_[i,1]>-0.15 else self.good_label for i in range(len(pc_))])
        # labels = (labels + self.correction) % 2
        
        # ==============
        # Strategy 2
        # =============
        # if self.ref_metric_scores is not None:
        #     labels_ = []
        #     for i in range(len(x)):
        #         label = self.clusterer_hdb.fit_predict(
        #             np.append(self.ref_metric_scores, np.append(pc_[i:i+1, -2:], score_[i:i+1].reshape(-1, 1), axis=1), axis=0)
        #         )
        #         labels_.append(1 if label[0]==-1 else 0)
        #     labels = np.array(labels_)
        
        # ===============
        # Strategy 3
        # ===============
        # pc_ = np.dot(pc_-np.mean(self.ref_pcs, axis=0, keepdims=True), self.matrix_multiplier) if self.matrix_multiplier is not None else pc_
        # score_ = score_ - np.median(self.ref_scores, axis=0, keepdims=True) if self.matrix_multiplier is not None else score_
        
        if self.custom_clusterer.configured:
            # print(colored('I am in the revising scores and labels.', 'light_red'))
            labels = self.custom_clusterer.predict(pc_[:, 0], pc_[:, 1])
            score_ = labels * score_
            
            # labels = np.ones_like(labels)
            # labels[(score_<0.15)] = 0
            
            labels[labels>=0.5] = 1
            labels[labels<0.5] = 0
        
        return pc_.astype(np.float32), score_.astype(np.float32), labels
    
    
    def validate_and_correct(self, x, y):
        _, _, label_clean = self.analyze(x, y)
        self.correction = 1 if np.mean(label_clean==self.bad_label) > 0.8 else self.correction
        return
    
    