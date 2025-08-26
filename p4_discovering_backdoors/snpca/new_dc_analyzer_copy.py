import torch
import numpy as np
import math
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from copy import deepcopy

from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


from _0_general_ML.model_utils.torch_model import Torch_Model
from _1_adversarial_ML.adversarial_attacks.all_available_adversarial_attacks import FGSM
from .fgsm_torch import FGSM_Torch

from ..model_utils.feature_activations import Feature_Activations

from utils_.general_utils import normalize
from utils_.visual_utils import show_image_grid
from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy, get_outputs, prepare_dataloader_from_tensor
from utils_.pca import PCA_of_SKLEARN



class ASNPCA_Stats:
    
    def __init__(
        self, 
        use_median: bool=True, 
        reference_activations: list[np.ndarray]=None,
    ):
        
        normalization_dict = {
            'values': [],
            'normalized_values': [],
            'max': [],
            'min': [],
            'median': []
        }
        
        self.reference_activations = reference_activations
        self.secret_analyzer = None
        self.secret_pcs = None # deepcopy(normalization_dict)
        
        self.use_median = use_median
        self.pca_analyzers = []
        
        self.pc_dict = deepcopy(normalization_dict)
        self.pc_p_dict = deepcopy(normalization_dict)
        self.pc_u_dict = deepcopy(normalization_dict)
        self.scores_dict = deepcopy(normalization_dict)
        self.pc_medians = []
        self.pcs_s = []
        self.scores_ = []
        
        return
    
    def normalize(self, values: np.ndarray, dict_name: dict, i: int=None):
        if i is None:
            dict_name['values'].append(values)
            dict_name['min'].append(np.min(values))
            dict_name['max'].append(np.max(values))
            dict_name['median'].append(np.median(values))
            normalized_value = (values - np.min(values)) / (np.max(values) - np.min(values))
            dict_name['normalized_values'].append(normalized_value)
            return normalized_value
        return (values - dict_name['min'][i]) / (dict_name['max'][i] - dict_name['min'][i])
    
    def get_feature(self, activations: np.ndarray, p_activations: np.ndarray, u_activations: np.ndarray=None, i: int=None, pca_analyzer: PCA_of_SKLEARN=None):
        
        if i is None:
            pc = pca_analyzer.transform(activations).reshape(len(activations), -1)
            pc_p = pca_analyzer.transform(p_activations).reshape(len(p_activations), -1)
            pc = self.normalize(pc, self.pc_dict)
            pc_p = self.normalize(pc_p, self.pc_p_dict)
            self.pca_analyzers.append(pca_analyzer)
        else:
            pc = self.pca_analyzers[i].transform(activations).reshape(len(activations), -1)
            pc_p = self.pca_analyzers[i].transform(p_activations).reshape(len(p_activations), -1)
            pc = self.normalize(pc, self.pc_dict, i=i)
            pc_p = self.normalize(pc_p, self.pc_p_dict, i=i)
            
        pcs_ = np.append(pc, pc_p, axis=1)
        if self.use_median:
            if i is None:
                pc_median = np.median(pcs_, axis=0, keepdims=True)
                self.pc_medians.append(pc_median)
            else:
                pc_median = self.pc_medians[i]
            pcs_ = pcs_ - pc_median
        if i is None:
            self.pcs_s.append(pcs_)
            
        return pcs_
    
    def get_pc_feature(self, activations: np.ndarray, p_activations: np.ndarray, u_activations: np.ndarray=None, i: int=None, pca_analyzer: PCA_of_SKLEARN=None):
        pcs_ = self.get_feature(activations, p_activations, i=i, pca_analyzer=pca_analyzer)
        if (i is not None) & (self.reference_activations is not None):
            pcs_ref = self.get_feature(self.reference_activations[0], self.reference_activations[1], i=i)
            pcs_ -= np.median(pcs_ref, axis=0, keepdims=True)
        return pcs_
    
    def update_stats(self, pca_analyzer: PCA_of_SKLEARN, activations: np.ndarray, p_activations: np.ndarray, u_activations: np.ndarray=None):
        
        pcs_ = self.get_pc_feature(activations, p_activations, pca_analyzer=pca_analyzer)
        
        scores_ = 0
        scores_ += np.mean(pcs_**2, axis=1)
        # scores_ += np.mean((activations-pca_analyzer.reconstruct(activations))**2, axis=1)
        scores_ = self.normalize(scores_, self.scores_dict)
        
        return pcs_, scores_
    
    def compute_pcs_and_score(self, activations: np.ndarray, p_activations: np.ndarray, i: int, u_activations: np.ndarray=None):
        
        pcs_ = self.get_pc_feature(activations, p_activations, i=i)
        
        scores_ = 0
        scores_ += np.mean(pcs_**2, axis=1)
        # scores_ += np.mean((activations-self.pca_analyzers[i].reconstruct(activations))**2, axis=1)
        scores_ = self.normalize(scores_, self.scores_dict, i=i)
        
        return pcs_, scores_
    
    
    
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
        radius_original = self.calculate_the_radius_from_normalized_ellipse(point_x, point_y)
        label = self.sigmoid(radius_original-1)
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
    
    
class PCA_Analyzer:
    
    def __init__(
        self, 
        model: Torch_Model, feature_model: Feature_Activations, 
        x, y, ac_, 
        epsilon=0.5,
        reference_loader=None,
        mode='one', sample_subset: int=30, iterations: int=100, std: float=0.2,
        threshold: float=0.3,
        verbose: bool=False,
        **kwargs
    ):
        
        # super().__init__()
        
        self.threshold = threshold
        
        self.model = model
        self.num_classes = model.data.num_classes
        self.feature_model = feature_model
        self.target_class = self.feature_model.target_class
        self.x = x
        self.y = y
        self.ac_ = ac_
        
        self.batch_size = self.model.model_configuration['batch_size']
        self.verbose = verbose
        
        self.attacker = FGSM(self.model, verbose=verbose)
        self.epsilon = epsilon
        self.std = std
        
        self.clusterer = KMeans(n_clusters=2, n_init='auto', init='k-means++')
        
        self.custom_clusterer = My_Clusterer()
        self.reference_loader = reference_loader
        self.pca_stats = ASNPCA_Stats(use_median=True)
        
        self.mode = mode
        if (self.mode == 'one') or (sample_subset is None):
            self.mode = 'one'
            self.iterations = 1
            self.sample_subset = len(self.ac_)
            self.replace = False
            self.prepare_analyzer_smooth(x, y)
        else:
            self.mode = 'multi'
            self.iterations = iterations
            self.sample_subset = sample_subset
            self.replace = True
            self.prepare_analyzer_smooth(x, y)
        
        return
    
    
    def np_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    
    def get_penulatimate_features(self, x, y):
        
        noise = np.random.normal(0, self.std*np.std(x), size=x.shape).astype(np.float32)
        x = x + noise
        xp = self.attacker.attack(x, self.target_class*np.ones_like(y), epsilon=self.epsilon, targeted=True, verbose=self.verbose)
        
        # _mnpca._model.mode = 'only_activations'
        activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        p_activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(xp, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        # _mnpca._model.mode = 'default'
            
        return activations, p_activations
    
    
    def prepare_clusterers(self):
        
        # =============
        # Cluster things
        # =============
        metric_scores = deepcopy(self.pcs_[:, -2:])
        self.metric_scores = np.append(metric_scores, self.scores_.reshape(-1, 1), axis=1)
        self.clusterer.fit(self.metric_scores)
        labels = self.clusterer.predict(self.metric_scores)
        
        # label_scores = [np.mean(labels==0), np.mean(labels!=0)]
        # self.good_label = np.argmax(label_scores); self.bad_label = np.argmin(label_scores)
        label_scores = [np.mean(self.scores_[labels==0]), np.mean(self.scores_[labels!=0])]
        self.good_label = np.argmin(label_scores); self.bad_label = np.argmax(label_scores)
        
        normalizing_max = np.max(self.pca_stats.pc_p_dict['max'])
        normalizing_min = np.min(self.pca_stats.pc_p_dict['min'])
        self.poisoning_score = np.abs(normalizing_max + normalizing_min)
        
        print(colored('\n***********************************************', 'light_green'))
        print(colored(f'***** Dataset: {self.model.data.data_name}, Class: {self.target_class}, Poisoning Score: {self.poisoning_score:.5f} ***********', 'light_green'))
        print(colored('***********************************************', 'light_green'))
        
        limited_x, limited_y = get_data_samples_from_loader(self.reference_loader, return_numpy=True)
        _limited_pa, _limited_scores, _limited_labels = self.analyze(limited_x, limited_y)
        limited_x, limited_y = limited_x[_limited_scores<0.2], limited_y[_limited_scores<0.2]
        limited_pa = []
        for i in range(50):
            _limited_pa, _limited_scores, _limited_labels = self.analyze(limited_x, limited_y)
            limited_pa.append(_limited_pa)
        self.limited_pa_ = np.concatenate(limited_pa, axis=0)
        self.limited_pa = np.median(limited_pa, axis=0)
        self.custom_clusterer.fit(self.limited_pa_)
        
        return
    
    
    def prepare_analyzer_smooth(self, x, y):
        
        activations, p_activations = self.get_penulatimate_features(x, y)
        
        for i in range(self.iterations):
            pca_analyzer = PCA_of_SKLEARN(self.ac_[np.random.choice(len(self.ac_), size=self.sample_subset, replace=self.replace)], n_components=1)
            pcs, scores = self.pca_stats.update_stats(pca_analyzer, activations, p_activations)
            
        # new thing
        pcs_s = []; scores_s = []
        for i in range(self.iterations):
            pcs, scores = self.pca_stats.compute_pcs_and_score(activations, p_activations, i)
            pcs_s.append(pcs); scores_s.append(scores)
        self.pcs_ = np.mean(pcs_s, axis=0)
        self.scores_ = np.mean(scores_s, axis=0)
        
        self.prepare_clusterers()
        
        return
    
    
    def pairwise_cosine_similarity_torch(self, flattened_clients_states):
        flattened_clients_states = torch.tensor(flattened_clients_states)
        normalized_input_a = torch.nn.functional.normalize(flattened_clients_states)
        res = torch.mm(normalized_input_a, normalized_input_a.T)
        res[res==0] = 1e-6
        return res.detach().cpu().numpy()
    
    
    def analyze(self, x, y, cluster_xy: bool=True, **kwargs):
        
        activations, p_activations = self.get_penulatimate_features(x, y)
        
        pcs_s = []; scores_s = []
        for i in range(self.iterations):
            pcs, scores = self.pca_stats.compute_pcs_and_score(activations, p_activations, i)
            pcs_s.append(pcs); scores_s.append(scores)
            
        pcs_s = np.mean(pcs_s, axis=0)
        scores_s = np.mean(scores_s, axis=0)
        metric_scores = deepcopy(pcs_s[:, -2:])
        metric_scores = np.append(metric_scores, scores_s.reshape(-1, 1), axis=1)
        
        label = self.good_label*np.ones((len(x)))
        if self.custom_clusterer.configured and cluster_xy:
            label = self.clusterer.predict(metric_scores)
            label[np.abs(pcs_s[:,1]) < 0.15] = self.good_label
            _label = self.custom_clusterer.predict(pcs_s[:, 0], pcs_s[:, 1])
            label[_label<0.5] = self.good_label
        
        return pcs_s, scores_s, label if self.poisoning_score>self.threshold else self.good_label*np.ones_like(label)
    
    
    def forward(self, x, y_out):
        
        y = np.argmax(y_out, axis=1)
        pc, score, label = self.analyze(x, y)
        
        vulnerable_ind = (y==self.target_class)
        vulnerable_ind = vulnerable_ind & (label==self.bad_label)
        
        y_mins = np.min(y_out, axis=1)
        y_random = y_out.copy()
        y_random[:, self.target_class] = y_mins
        # y_random = np.random.normal(0, self.num_classes, size=y_out.shape)
        # y_random[y_random==self.target_class] += 1
        # y_random = y_random % self.num_classes
        y_out[vulnerable_ind] = y_random[vulnerable_ind]
        
        return y_out
    
    
    def __forward(self, x, y_out):
        
        y = np.argmax(y_out, axis=1)
        pc, score, label = self.analyze(x, y)
        
        vulnerable_ind = (y==self.target_class)
        vulnerable_ind = vulnerable_ind & (label==self.bad_label)
        
        y_random = np.random.normal(0, self.num_classes, size=y_out.shape)
        y_random[y_random==self.target_class] += 1
        y_random = y_random % self.num_classes
        y_out[vulnerable_ind] = y_random[vulnerable_ind]
        
        return y_out
    
    
    