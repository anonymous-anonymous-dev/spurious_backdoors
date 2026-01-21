import torch
import numpy as np
import math
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
from scipy.stats import t as t_test, f as ellipse_test
import gc

from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


from _0_general_ML.data_utils.datasets import GTSRB
from ..utils.one_channel_data import Channel1_Dataset, Channel1_Torch_Dataset, Custom_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model
from _0_general_ML.model_utils.generalized_model_activations_wrapper import Dependable_Feature_Activations
from _1_adversarial_ML.adversarial_attacks.all_available_adversarial_attacks import FGSM, PGD, Universal_Adversarial_Perturbation
from .fgsm_torch import FGSM_Torch
from .image_processor import Image_PreProcessor

from ...model_utils.feature_activations import Feature_Activations

from utils_.general_utils import normalize, exponential_normalize
from utils_.visual_utils import show_image_grid
from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy, get_outputs, prepare_dataloader_from_tensor
from utils_.pca import PCA_of_SKLEARN
from utils_.lda import LDA2Class



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
    
    def get_feature(self, activations: np.ndarray, p_activations: np.ndarray=None, u_activations: np.ndarray=None, i: int=None, pca_analyzer: PCA_of_SKLEARN=None):
        
        if i is None:
            pc = pca_analyzer.transform(activations).reshape(len(activations), -1)
            pc = self.normalize(pc, self.pc_dict)
            self.pca_analyzers.append(pca_analyzer)
            pc_median = np.median(pc, axis=0, keepdims=True)
            self.pc_medians.append(pc_median)
        else:
            pc = self.pca_analyzers[i].transform(activations).reshape(len(activations), -1)
            pc = self.normalize(pc, self.pc_dict, i=i)
            pc_median = self.pc_medians[i]
        if self.use_median:
            pc -= pc_median
        if i is None:
            self.pcs_s.append(pc)
            
        return pc
    
    def get_pc_feature(self, activations: np.ndarray, p_activations: np.ndarray=None, u_activations: np.ndarray=None, i: int=None, pca_analyzer: PCA_of_SKLEARN=None):
        pcs_ = self.get_feature(activations, i=i, pca_analyzer=pca_analyzer)
        if (i is not None) & (self.reference_activations is not None):
            pcs_ref = self.get_feature(self.reference_activations[0], self.reference_activations[1], i=i)
            pcs_ -= np.median(pcs_ref, axis=0, keepdims=True)
        return pcs_
    
    def get_score(self, pcs_: np.ndarray):
        return np.sum(pcs_**2, axis=1)
    
    def update_stats(self, pca_analyzer: PCA_of_SKLEARN, activations: np.ndarray, p_activations: np.ndarray=None, u_activations: np.ndarray=None):
        
        pcs_ = self.get_pc_feature(activations, pca_analyzer=pca_analyzer)
        
        scores_ = 0
        scores_ += self.get_score(pcs_) #np.mean(pcs_**2, axis=1)
        # scores_ += np.mean((activations-pca_analyzer.reconstruct(activations))**2, axis=1)
        # scores_ = self.normalize(scores_, self.scores_dict)
        
        return pcs_, scores_
    
    def compute_pcs_and_score(self, activations: np.ndarray, i: int, p_activations: np.ndarray=None, u_activations: np.ndarray=None):
        
        pcs_ = self.get_pc_feature(activations, i=i)
        
        scores_ = 0
        scores_ += self.get_score(pcs_) #np.mean(pcs_[:, 0]**2, axis=1)
        # scores_ += np.mean((activations-self.pca_analyzers[i].reconstruct(activations))**2, axis=1)
        # scores_ = self.normalize(scores_, self.scores_dict, i=i)
        
        return pcs_, scores_
    
    
    
class My_Clusterer_Original:
    
    def __init__(self, hardness: float=20, std_multiplier: float=3):
        self.configured = False
        self.circle_configured = False
        self.center_lies_below_the_learned_tanged = None
        self.hardness = hardness
        self.std_multiplier = std_multiplier
        self.lr = LinearRegression()
        return
    
    
    def sigmoid(self, value: float):
        return 1/(1+np.exp(-self.hardness * value))
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
        width = self.std_multiplier * np.std(new_multiplied[:,0])
        height = self.std_multiplier * np.std(new_multiplied[:,1])
        
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
        
        if self.width==0 or self.height==0:
            assert False, "Width and Height of the ellipse cannot be zero."
        
        return
    
    
    def readjust_center(self, new_center):
        self.center_x = new_center[0]
        self.center_y = new_center[1]
        self.center = np.array([self.center_x, self.center_y]).reshape(1,2)
        self.right_most_point = np.dot(np.array([self.width/2, 0]).reshape(1, 2), self.inverse_matrix_multiplier) + self.center
        self.left_most_point = np.dot(np.array([-self.width/2, 0]).reshape(1, 2), self.inverse_matrix_multiplier) + self.center
        if self.right_most_point[0,0]<self.left_most_point[0,0]:
            right_most_point = self.left_most_point.copy()
            self.left_most_point = self.right_most_point.copy()
            self.right_most_point = right_most_point.copy()
        self.ellipse_slope = (self.right_most_point[0,1]-self.left_most_point[0,1])/(self.right_most_point[0,0]-self.left_most_point[0,0])
        self.intercept = self.left_most_point[0,1] - self.ellipse_slope*self.left_most_point[0,0]
        return
    
    
    def __find_ellipse_tangent(self, x0, y0):
        """
        Finds the equation of the tangent line to an ellipse at a given point.

        Args:
            x0 (float): x-coordinate of the point on the ellipse.
            y0 (float): y-coordinate of the point on the ellipse.
            a (float): Length of the semi-major axis.
            b (float): Length of the semi-minor axis.
            h (float, optional): x-coordinate of the ellipse's center. Defaults to 0.
            k (float, optional): y-coordinate of the ellipse's center. Defaults to 0.

        Returns:
            tuple: A tuple containing (slope, intercept) of the tangent line.
        """
        
        a = self.width
        b = self.height
        h = self.center_x
        k = self.center_y
        
        # Calculate the slope of the tangent at (x0, y0)
        # Handle division by zero if y0 == k and a non-vertical tangent is expected
        if np.isclose(y0, k) and not np.isclose(x0, h): # Horizontal tangent at (x0, k)
            slope = 0.0
        elif np.isclose(x0, h) and not np.isclose(y0, k): # Vertical tangent at (h, y0)
            # Slope is undefined, representing a vertical line x = x0
            return np.inf, x0 # Return infinity for slope and x-intercept
        elif np.isclose(a, 0) or np.isclose(b, 0): # Degenerate ellipse
            raise ValueError("Semi-axes 'a' and 'b' must be non-zero.")
        else:
            slope = - (b**2 * (x0 - h)) / (a**2 * (y0 - k))

        # Calculate the y-intercept
        intercept = y0 - slope * x0

        return slope, intercept
    
    
    def find_ellipse_tangent(self, px, py):
        """
        Calculates the slope (m) and y-intercept (c) of the tangent line 
        to a rotated ellipse at a specific point (px, py).

        Args:
            center_x, center_y: Center coordinates of the ellipse.
            width, height: Full width and height of the ellipse.
            angle_degrees: Rotation angle of the ellipse in degrees.
            px, py: The point on the ellipse where the tangent is needed.

        Returns:
            A tuple (slope, intercept) of the tangent line in the form y = slope * x + intercept.
            Returns (None, px) if the tangent is vertical (undefined slope).
        """
        
        width = self.width
        height = self.height
        center_x = self.center_x
        center_y = self.center_y
        angle_degrees = self.angle_degrees
        
        # 1. Convert angle to radians for NumPy and get semi-axes
        angle_rad = np.deg2rad(angle_degrees)
        a = width / 2.0  # semi-major or semi-minor axis
        b = height / 2.0 # semi-major or semi-minor axis

        # 2. Translate point and center to the origin
        # This transforms the point (px, py) to (Tx, Ty) relative to the origin
        Tx = px - center_x
        Ty = py - center_y

        # 3. Inverse Rotate the translated point to align with axes
        # We rotate by -angle_rad
        cos_angle = np.cos(-angle_rad)
        sin_angle = np.sin(-angle_rad)
        
        # The point in the standard (non-rotated) ellipse coordinate system (X, Y)
        X = Tx * cos_angle - Ty * sin_angle
        Y = Tx * sin_angle + Ty * cos_angle

        # 4. Calculate the slope in the standard system
        # Equation: X^2/a^2 + Y^2/b^2 = 1. Derivative dY/dX = - (b^2 * X) / (a^2 * Y)
        if abs(Y) < 1e-9:
            # Vertical tangent in the standard system, which becomes a rotated vertical tangent
            slope_std = None
        else:
            slope_std = - (b**2 * X) / (a**2 * Y)
            
        # 5. Transform the slope back to the original coordinate system
        # The transformation formula for slopes under rotation (from Stack Overflow link):
        # m_orig = (m_std + tan(angle_rad)) / (1 - m_std * tan(angle_rad))
        tan_angle = np.tan(angle_rad)

        if slope_std is None:
            # If std tangent is vertical, orig tangent slope is -cot(angle_rad)
            if abs(tan_angle) < 1e-9:
                # If angle is 0 or 180 deg, tangent stays vertical
                final_slope = None
            else:
                final_slope = -1.0 / tan_angle
        else:
            # Apply the slope rotation formula
            denominator = 1 - slope_std * tan_angle
            if abs(denominator) < 1e-9:
                # Vertical tangent in the original system
                final_slope = None
            else:
                final_slope = (slope_std + tan_angle) / denominator

        # 6. Calculate the intercept using the point-slope form
        # y - py = m * (x - px)  =>  y = m*x + (py - m*px)
        if final_slope is None:
            # For a vertical line, we return the x-intercept (which is px)
            final_intercept = px
        else:
            final_intercept = py - final_slope * px

        return final_slope, final_intercept
    
    
    def move_this_point_to_ellipse_boundary(self, point_x, point_y, n_points_to_check: int=100):
        
        x_intervals = (point_x-self.center_x)/n_points_to_check; y_intervals = (point_y-self.center_y)/n_points_to_check
        multiple_x_values = np.arange(self.center_x, point_x+x_intervals, x_intervals)
        multiple_y_values = np.arange(self.center_y, point_y+y_intervals, y_intervals)
        
        closest_x, closest_y = None, None
        for i in range(n_points_to_check):
            label = self.predict(multiple_x_values[i], multiple_y_values[i])
            if label>0.5:
                closest_x = multiple_x_values[i]
                closest_y = multiple_y_values[i]
                break
            
        return closest_x, closest_y
    
    
    def fit_tangent(self, ood_points: np.ndarray):
        
        point_x = ood_points[0]
        point_y = ood_points[1]
        
        line_slope = (self.center_y-point_y)/(self.center_x-point_x)
        line_y_intercept = self.center_y - line_slope*self.center_x
        
        if self.predict(point_x, point_y, use_tangent=False)<0.5:
            assert False
        if self.predict(point_x+point_x/10, point_y, use_tangent=False)<0.5:
            assert False
        
        closest_x, closest_y = self.move_this_point_to_ellipse_boundary(point_x, point_y, n_points_to_check=200)
        
        # self.tangent_m, self.tangent_c = self.find_ellipse_tangent(closest_x, closest_y)
        self.tangent_m = -1*(self.center_x-closest_x)/(self.center_y-closest_y)
        self.tangent_c = closest_y - self.tangent_m*closest_x
        
        # ly = self.tangent_m*self.left_most_point[0,0] + self.tangent_c
        # ry = self.tangent_m*self.right_most_point[0,0] + self.tangent_c
        # self.points_for_tangent = np.array([[self.left_most_point[0,0], ly],[self.right_most_point[0,0], ry]])
        
        diff_x = np.abs(self.left_most_point[0,0]-self.right_most_point[0,0])
        x_points = np.arange(closest_x-diff_x, closest_x+diff_x, diff_x/20)
        y_points = self.tangent_m * x_points + self.tangent_c
        
        x_points = x_points[np.abs(y_points-closest_y)<2*self.height]
        y_points = y_points[np.abs(y_points-closest_y)<2*self.height]
        y_points = y_points[np.abs(x_points-closest_x)<2*self.width]
        x_points = x_points[np.abs(x_points-closest_x)<2*self.width]
        
        # lx, rx = closest_x-diff_x, closest_x+diff_x
        # ly = self.tangent_m*lx+self.tangent_c
        # ry = self.tangent_m*rx+self.tangent_c
        self.points_for_tangent = np.stack([x_points,y_points], axis=1)
        
        predicted_y = self.tangent_m*self.center_x+self.tangent_c
        if predicted_y>=self.center_y: self.center_lies_below_the_learned_tanged = True
        else: self.center_lies_below_the_learned_tanged = False
        
        # self.circle_radius = ((closest_x-self.center_x)**2 + (closest_x-self.center_y)**2)**0.5
        # self.center_x_of_circle = closest_x - 2*(closest_x-self.center_x)/self.circle_radius
        # self.center_y_of_circle = closest_x - 2*(closest_y-self.center_y)/self.circle_radius
        # self.circle_radius *= 2
        
        self.circle_configured = True
        
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
    
    
    def predict(self, point_x, point_y, use_tangent: bool=True):
        
        # if point_x and point_y are not arrays, but floats, convert them to arrays
        if not isinstance(point_x, np.ndarray):
            point_x = np.array([point_x])
        if not isinstance(point_y, np.ndarray):
            point_y = np.array([point_y])
        
        radius_original = self.calculate_the_radius_from_normalized_ellipse(point_x, point_y)
        label = self.sigmoid(radius_original-1)
        
        if (self.center_lies_below_the_learned_tanged is not None) and (use_tangent):
            predicted_y = self.tangent_m*point_x + self.tangent_c
            label_2 = self.sigmoid(point_y-predicted_y if self.center_lies_below_the_learned_tanged else predicted_y-point_y)
            # label = np.min(np.stack([label, label_2], axis=1).reshape(-1, 2), axis=1)
            label[(label_2<=0.5) & (label>0.5)] = label_2[(label_2<=0.5) & (label>0.5)]
            # if self.center_lies_below_the_learned_tanged:
            #     label[predicted_y>point_y] = 0.1
            # else:
            #     label[predicted_y<point_y] = 0.1
        
        return label
    
    
class My_Clusterer:
    
    def __init__(self, iterations: int=1, hardness: float=20, std_multiplier: float=3):
        self.configured = False
        self.circle_configured = False
        self.iterations = iterations
        self.std_multiplier = std_multiplier
        self.hardness = hardness
        self.ellipses = [My_Clusterer_Original(hardness=self.hardness, std_multiplier=self.std_multiplier)] * self.iterations
        self.sklearn_clusterer = KMeans(n_clusters=2, n_init='auto', init='k-means++')
        self.good_sklearn_label = -1
        return
    
    def fit(self, limited_pa: np.ndarray):
        self.iterations = np.clip(len(limited_pa)//500, 1, self.iterations) # self.iterations = min(iterations, self.iterations)
        for i in range(self.iterations):
            _limited_pa = limited_pa[np.random.choice(len(limited_pa), size=min(500, len(limited_pa)), replace=False)]
            self.ellipses[i].fit(_limited_pa)
        self.configured = True
        return
    
    def fit_sklearn(self, tpa: np.ndarray):
        tlabels_sklearn = self.sklearn_clusterer.fit_predict(tpa)
        tlabels_custom = self.predict(tpa[:,0], tpa[:,1])
        self.good_sklearn_label = 0 if np.median(tlabels_sklearn[tlabels_custom<0.5])<0.5 else 1
        return
    
    def readjust_center(self, new_center, weightage: float=0.5):
        weightage = np.clip(weightage, 0., 1.)
        for i in range(self.iterations):
            self.ellipses[i].readjust_center(weightage*new_center+(1-weightage)*self.ellipses[i].center.reshape(-1))
        return
    
    def fit_tangent(self, ood_pa: np.ndarray):
        try:
            for i in range(self.iterations):
                self.ellipses[i].fit_tangent(ood_pa)
            self.circle_configured = True
        except:
            print(f'Could not fit tangent.')
        return
    
    def predict(self, point_x, point_y, use_tangent: bool=True):
        labels = []
        for i in range(self.iterations):
            labels.append(self.ellipses[i].predict(point_x, point_y, use_tangent=use_tangent))
        labels = np.array(labels)
        labels = np.mean(labels, axis=0)
        
        if self.good_sklearn_label != -1:
            labels_sklearn = self.sklearn_clusterer.predict(np.stack([point_x, point_y], axis=1).reshape(len(point_x), 2))
            labels[np.where(labels_sklearn==self.good_sklearn_label)] = 0.1
            
        return labels
    
    def update_std_multiplier(self, new_std_multiplier: float=None):
        for ellipse in self.ellipses:
            ellipse.configure(ellipse.center_x, ellipse.center_y, new_std_multiplier*ellipse.width/ellipse.std_multiplier, new_std_multiplier*ellipse.height/ellipse.std_multiplier, ellipse.angle_degrees)
            ellipse.std_multiplier = new_std_multiplier
        return
    
    
class Cosine_Similarity_PCA_Clusterer_for_Target_Class:
    
    def __init__(self, target_class: int=0, ac_ref_dict: dict=None, ac_nonref_dict: dict=None, batch_size: int=16, mode: str='train', verbose: bool=False):
        
        self.target_class = target_class
        self.ac_ref_dict = deepcopy(ac_ref_dict)
        # self.keys = list(self.ac_ref_dict.keys())
        size_of_ref = len(self.ac_ref_dict)
        self.batch_size = batch_size
        
        self.clusterer = KMeans(n_clusters=2, n_init='auto', init='k-means++')
        self.custom_clusterer = My_Clusterer(hardness=1, std_multiplier=3 if size_of_ref>=30 else 4)
        self.cluster_threshold = 0.5
        
        # self.replace = False
        # self.disable_kmeans_cluster = False
        self.good_label = 0; self.bad_label = 1
        self.poisoning_score = 1
        self.adaptive_thresh = 0.5; self.ratio = 0; self.alpha_ratio = 0
        self.mode = mode
        
        self.verbose = verbose
        
        _, self.ac_ref = self.get_pc_features(self.ac_ref_dict, use_for_referencing=True)
        ac_non_ref_ = self.get_pc_features(ac_nonref_dict)[1] if ac_nonref_dict is not None else self.ac_ref[:1]
                
        size_of_random_ref = 1000
        random_ref = np.random.choice(np.arange(0.51, 0.81, 0.01), size=size_of_random_ref*self.ac_ref.shape[1], replace=True).reshape(-1, self.ac_ref.shape[1])
        random_ref = random_ref * np.random.choice([-1, 1], size=size_of_random_ref, replace=True).reshape(-1, 1)
        self.pca_sklearn = PCA_of_SKLEARN(np.concatenate([self.ac_ref, ac_non_ref_ if ac_nonref_dict is not None else random_ref], axis=0), n_components=2, mean_centric=False)
        self.pcs_ref = self.pca_sklearn.transform(self.ac_ref)
        # Computing the weight of each reference sample based on its nearness to other reference samples (only when large number of reference samples are available)
        weight_of_each = []
        for i, value_set in enumerate(self.pcs_ref):
            weight_ = np.sum(np.square(value_set.reshape(1, -1) - self.pcs_ref), axis=1)
            weight_ = np.mean(np.sort(weight_)[:3])
            weight_of_each.append(weight_)
        weight_of_each = np.array(weight_of_each)
        weight_of_each = exponential_normalize(-100*weight_of_each).reshape(-1, 1)
        self.pcs_ref_median = np.sum(weight_of_each*self.pcs_ref, axis=0, keepdims=True)
        
        self.scores_ = np.mean((self.pcs_ref-self.pcs_ref_median)**2, axis=1)
        self.prepare_clusterers()
        
        return
    
    
    def adjust_with_train_features(self, train_x_dict: dict):
        tpa, _, tpa_label, tpa_probabilities = self.analyze(train_x_dict)
        self.custom_clusterer.fit_sklearn(tpa)
        print(colored(f'[PCA_Analyzer] Updated the sklearn clusterer with the training data', 'light_red'))
        return
    
    
    def adjust_with_ood_features(self, ood_x_dict: dict, adv_ood_x_dict: dict=None):
        
        opa, _, opa_label, _ = self.analyze(ood_x_dict, use_tangent=False)
        adv_opa, _, adv_opa_label, _ = self.analyze(adv_ood_x_dict, use_tangent=False)
        condition = False
        condition = condition or (self.bad_label in opa_label)
        condition = condition or (self.bad_label in adv_opa_label)
        if condition:
            mean_opa = np.mean(adv_opa[adv_opa_label==self.bad_label], axis=0)
            if (self.bad_label in opa_label): 
                mean_opa = 0.5*np.mean(opa[opa_label==self.bad_label], axis=0) + 0.5*mean_opa
                # mean_opa = np.mean(opa[opa_label==self.bad_label], axis=0)
            self.custom_clusterer.fit_tangent(mean_opa)
        self.adv_opa = adv_opa.copy()
        self.opa = opa.copy()
        
        return self.opa, self.adv_opa
    
    
    def prepare_clusterers(self):
        self.limited_pa = self.pcs_ref-self.pcs_ref_median
        self.custom_clusterer.fit(self.limited_pa)
        return
    
    
    def sigmoid(self, value: float, z: float=1):
        return 1/(1+np.exp(-z * value))
    def analyze(self, x_ref_dict, cluster_xy: bool=True, use_tangent: bool=True, **kwargs):
        
        activations, p_activations = self.get_pc_features(x_ref_dict)
        
        pcs_s = self.pca_sklearn.transform(activations) - self.pcs_ref_median
        scores_s = np.mean(pcs_s**2, axis=1)
        # scores_s = self.mahalanobis_distance(pcs_s)
        
        label = self.good_label*np.ones((len(activations)))
        _label = self.good_label*np.ones((len(activations)))
        if self.custom_clusterer.configured and cluster_xy:
            _label = self.custom_clusterer.predict(pcs_s[:, 0], pcs_s[:, 1], use_tangent=use_tangent)
            # label[_label<0.5] = self.good_label
            
            self.adaptive_thresh = 0.5
            if self.mode == 'test':
                thresh_max = 0.85; tolerable_level_max = 0.1; best_level_max = 0
                thresh_min = 0.5; level_max = best_level_max; level_min = tolerable_level_max
                self.ratio = np.mean(self.sigmoid(_label-thresh_max, z=1000))
                self.adaptive_thresh = (thresh_max-thresh_min) * self.sigmoid(tolerable_level_max-self.ratio, z=1000) + thresh_min
                # slope = (thresh_max-thresh_min)/(level_max-level_min)
                # intercept = thresh_max - slope * level_max
                # # self.ratio = 0.35 if np.mean(self.sigmoid(_label-0.6, z=10000))>0.1 else self.ratio
                # # self.ratio = self.alpha_ratio*self.ratio + (1-self.alpha_ratio)*value
                # # self.alpha_ratio
                # self.adaptive_thresh = min(max(slope*self.ratio+intercept, thresh_min), thresh_max)
                # # _label -= (self.adaptive_thresh-0.5)
                # # print(f'Adaptive thresholding.')
            
            label = _label.copy()
            label[_label<=self.adaptive_thresh] = self.good_label
            label[_label>self.adaptive_thresh] = self.bad_label
        else:
            print(colored(f'Not using clusterer because it is not configured.', 'light_red'))
        
        return pcs_s, scores_s, label, _label if self.custom_clusterer.configured else label
    
    
    def get_pc_features(self, _p_activations, use_for_referencing: bool=False, batchwise: bool=True):
        
        def pairwise_cosine_similarity_torch_one_batch(state_1, state_2):
            if isinstance(state_1, np.ndarray): state_1 = torch.tensor(state_1).to('cuda')
            if isinstance(state_2, np.ndarray): state_2 = torch.tensor(state_2).to('cuda')
            normalized_input_a = torch.nn.functional.normalize(state_1)
            normalized_input_b = torch.nn.functional.normalize(state_2)
            res = torch.mm(normalized_input_a, normalized_input_b.T)
            res[res==0] = 1e-6
            return res.detach().cpu().numpy()
        
        def pairwise_cosine_similarity_torch(state_1, state_2):
            max_size = max(len(state_1), len(state_2)) if len(state_1)!=len(state_2) else len(state_1)
            
            n_batches = max_size // self.batch_size
            n_batches += 1 if (n_batches*self.batch_size)<max_size else 0
            
            cs_values = []
            for i in range(n_batches):
                _cs = pairwise_cosine_similarity_torch_one_batch(
                    state_1[i*self.batch_size:(i+1)*self.batch_size], 
                    state_2[i*self.batch_size:(i+1)*self.batch_size] if len(state_2)==len(state_1) else state_2
                )
                _cs = _cs[np.arange(len(_cs)), np.arange(len(_cs))] if len(state_2)==len(state_1) else _cs
                cs_values.append(_cs)
            
            assert len(cs_values) > 0, f'Len of the list cannot be less than 1.'
            cs_values = np.concatenate(cs_values, axis=0) if len(cs_values)>1 else cs_values[0]
            
            return cs_values
        
        def get_feature_from_dict(features_dict: dict, ref_dict: dict=None):
            flattened_ref = self.ac_ref_dict if ref_dict is None else ref_dict
            flattened_dict = features_dict
            
            outputs, outputs_mag, outputs_2 = [], [], []
            for i in range(flattened_ref.shape[1]):
                outputs.append(pairwise_cosine_similarity_torch(flattened_dict[:,i], np.median(flattened_ref[:,i], axis=0, keepdims=True)))
                if i != 0:
                    value = pairwise_cosine_similarity_torch(flattened_dict[:,i], flattened_dict[:,0])
                    outputs_2.append(value)
                # mag = 1/(1+get_norm(flattened_dict[:, i], flattened_ref[:, i]))
                # outputs_mag.append(2*mag-1)
                torch.cuda.empty_cache(); gc.collect()
            outputs += [np.stack(outputs_2, axis=1)] if len(outputs_2)>0 else []
            # outputs += [np.stack(outputs_mag, axis=1)] if len(outputs_mag)>0 else []
            return np.concatenate(outputs, axis=1) if len(outputs)>1 else np.concatenate(outputs+outputs, axis=1)
        
        activations = get_feature_from_dict(_p_activations, ref_dict=_p_activations if use_for_referencing else None)
        
        if use_for_referencing:
            return deepcopy(_p_activations), activations.copy()
        
        return activations, activations.copy()
    
    
    
class PCA_Analyzer_Universal_Efficient:
    
    def __init__(
        self, 
        model: Torch_Model, fm,
        train_x=None, train_y=None,
        reference_loader=None,
        target_classes: list[int]=[0, 1, 2],
        epsilon: float=0.5, std: float=0.,
        e_t: float=0.1,
        verbose: bool=False,
        visible_spuriousity_only: bool=False,
        **kwargs
    ):
        
        def inverse_adversarial_training(model_torch: Torch_Model, im: Image_PreProcessor, epochs: int=2):
            model_torch.freeze_last_n_layers(n=None)
            model_torch.unfreeze_last_n_layers(n=7)
                
            batch_size=model_torch.model_configuration['batch_size']; dl = torch.utils.data.DataLoader(model_torch.data.train, batch_size=batch_size)
            x, y = get_data_samples_from_loader(dl, return_numpy=True)
            attack_adv = FGSM(model_torch)
            # adv_x = attack_adv.attack(x, y, epsilon=0.2, targeted=True, iterations=50)
            adv_y = np.argmax(get_outputs(model_torch.model, prepare_dataloader_from_numpy(x, x, batch_size=batch_size), return_numpy=True), axis=1)
            attack_uap = attack_adv# Universal_Adversarial_Perturbation(model_torch);
            for ep in range(epochs):
                uap_x = attack_uap.attack(x, adv_y, epsilon=0.2, iterations=50, targeted=False)
                uap_y = np.argmax(get_outputs(model_torch.model, prepare_dataloader_from_numpy(uap_x, uap_x, batch_size=batch_size), return_numpy=True), axis=1)
                new_dl = prepare_dataloader_from_numpy(np.concatenate((x, uap_x), axis=0), np.concatenate((adv_y, uap_y), axis=0), batch_size=batch_size, shuffle=True)
                model_torch.train_shot(new_dl, ep)
            return
        
        # super().__init__()
        
        self.e_t = e_t
        
        # Classical image processor for processing images using classical image processing techniques
        self.image_processor = Image_PreProcessor(batch_size=None) #self.model.model_configuration['batch_size'])
        
        self.model_for_backup = model
        self.model = Torch_Model(model.data, model.model_configuration, path=model.path)
        self.model.model.load_state_dict(model.model.state_dict())
        inverse_adversarial_training(self.model, self.image_processor)
        
        self.num_classes = model.data.num_classes
        self.target_classes = target_classes if target_classes is not None else np.arange(model.data.num_classes).astype('int')
        
        # Preparing the feature extractor to extract the features of the the last 15 layers
        len_of_layers = len(self.model.get_children())
        num_layers = min((len_of_layers//2)+1, 15) # if not visible_spuriousity_only else 10)
        self.layer_numbers = np.arange(-num_layers, 0, 1)
        
        print(colored(f'[PCA Analyzer] layer_numbers: {self.layer_numbers}', 'red'))
        self.general_feature_model = Dependable_Feature_Activations(self.model, layer_numbers=self.layer_numbers, get_weighted_activations_for_the_last_layer=True, output_type='dict')
        uap_attack = Universal_Adversarial_Perturbation(self.model)
        
        self.batch_size = self.model.model_configuration['batch_size']
        self.verbose = verbose
        
        # Setting some attack-related variables
        self.attacker = FGSM(self.model)
        self.epsilon = epsilon
        self.std = std
        
        self.good_label = 0; self.bad_label = 1
        self.poisoning_score = 1
        self.visible_spuriousity_only = visible_spuriousity_only
        
        self.ref_x, self.ref_y = get_data_samples_from_loader(reference_loader, return_numpy=True)
        # self.ac_ref_dict = self.get_adversarial_penultimate_features(self.ref_x, self.ref_y)
        print(colored(f'[PCA_Analyzer] Reference activations collected from {len(self.ref_x)} samples.', "red"))
        
        self.untargeted_uaps, self.targeted_uaps = {}, {}
        self.feature_models = {}
        for t, _target_class in enumerate(self.target_classes):
            fm = Dependable_Feature_Activations(self.model, layer_numbers=self.layer_numbers, get_weighted_activations_for_the_last_layer=True, target_class=_target_class, output_type='dict')
            fm.model.eval()
            self.feature_models[_target_class] = fm
            
            adv_ref_x = uap_attack.attack(self.ref_x, _target_class*np.ones_like(self.ref_y), epsilon=0.1, iterations=10, targeted=False)
            self.untargeted_uaps[_target_class] = np.mean(adv_ref_x-self.ref_x, axis=0, keepdims=True)
            adv_ref_x = uap_attack.attack(self.ref_x, _target_class*np.ones_like(self.ref_y), epsilon=0.1, iterations=10, targeted=True)
            self.targeted_uaps[_target_class] = np.mean(adv_ref_x-self.ref_x, axis=0, keepdims=True)
        
        # OOD Analysis for learning tangent
        ood_data = GTSRB(preferred_size=self.model.data.preferred_size, data_means=self.model.data.data_means, data_stds=self.model.data.data_stds)
        if 'mnist' in self.model.model_configuration['dataset_name']:
            ood_data = Channel1_Torch_Dataset(ood_data)
        ood_data = Custom_Dataset(ood_data, train_size=None, max_target=self.model.data.num_classes-1)
        ood_x, ood_y = get_data_samples_from_loader(torch.utils.data.DataLoader(ood_data.train, batch_size=self.model.model_configuration['batch_size'], shuffle=True), return_numpy=True, size=100)
        
        at = PGD(self.model, verbose=verbose)
        
        self.deciders = []
        for t, _target_class in enumerate(self.target_classes):
            print('\r', colored(f'Working on class {_target_class}, [{t+1}/{len(self.target_classes)}]', 'green'))
            
            # this_dict = {l: self.ac_ref_dict[l][self.ref_y==_target_class] for l in self.ac_ref_dict.keys()}
            this_dict = self.get_adversarial_penultimate_features(self.ref_x[self.ref_y==_target_class], self.ref_y[self.ref_y==_target_class])
            that_dict = None
            # if np.sum(self.ref_y!=_target_class)>0:
            #     non_ref_x = self.ref_x[self.ref_y!=_target_class]; non_ref_x = non_ref_x[np.random.choice(len(non_ref_x), min(len(non_ref_x), len(this_dict)), replace=False)]
            #     that_dict = self.get_adversarial_penultimate_features(non_ref_x, _target_class*np.ones((len(non_ref_x))).astype('int'))
            
            dcdr = Cosine_Similarity_PCA_Clusterer_for_Target_Class(
                _target_class, this_dict,
                batch_size=self.batch_size,
                verbose=verbose
            )
            self.deciders.append(dcdr)

            if not self.visible_spuriousity_only:
                self.use_training_data_for_references = True
                self.use_training_data_for_references = self.use_training_data_for_references and (train_x is not None)
                self.use_training_data_for_references = self.use_training_data_for_references and (train_y is not None)
                if self.use_training_data_for_references:
                    self.train_x, self.train_y = train_x[train_y==_target_class], train_y[train_y==_target_class]
                    if len(self.train_x) > 30:
                        train_x_dict = self.get_adversarial_penultimate_features(self.train_x, self.train_y)
                        dcdr.adjust_with_train_features(train_x_dict)
                
                adv_ood_x = at.attack(ood_x, _target_class*np.ones_like(ood_y), epsilon=0.05, iterations=10); 
                adv_ood_y = _target_class*np.ones_like(ood_y)
                ood_x_dict = self.get_adversarial_penultimate_features(ood_x, adv_ood_y)
                adv_ood_x_dict = self.get_adversarial_penultimate_features(adv_ood_x, adv_ood_y)
                opa, adv_opa = dcdr.adjust_with_ood_features(ood_x_dict, adv_ood_x_dict)
                if _target_class==0:
                    self.opa, self.adv_opa = opa, adv_opa
                    
                dcdr.mode = 'test'
        
        return
    
    
    def np_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    
    def get_adversarial_penultimate_features(self, x, y):
        
        assert np.mean(np.array([k in self.target_classes for k in np.unique(y)]))==1, f'Undefended classes found.'
        
        # x = self.image_processor.local_random_skew(x)
        final_activations = None
        for t, _target_class in enumerate(self.target_classes):
            if _target_class in y:
                ref_dict_for_y = self.get_adversarial_penultimate_features_fm(x[y==_target_class], y[y==_target_class], self.feature_models[_target_class])
                if final_activations is None:
                    final_activations = np.zeros([len(x)]+list(ref_dict_for_y[0].shape))
                final_activations[y==_target_class] = ref_dict_for_y
                
        if final_activations is None: assert False, 'No suitable data found.'
            
        return final_activations
    
    
    def get_adversarial_penultimate_features_fm(self, x, y, fm: Dependable_Feature_Activations):
        
        def dict_to_flattened_np_array(features_dict: dict, axis: int=2):
            return np.concatenate([features_dict[layer_num] for layer_num in features_dict.keys()], axis=axis)
        
        activations = get_outputs(fm, prepare_dataloader_from_numpy(x, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        
        x_max = max(1, np.max(x))
        x_min = min(0, np.min(x))
        
        all_activations = []
        xp = self.attacker.attack(x, y, epsilon=self.epsilon, targeted=True, verbose=self.verbose) # this one works well, we have seen it
        # xp = self.attacker.attack(x+self.untargeted_uaps[y[0]], y, epsilon=self.epsilon, targeted=True, verbose=self.verbose) # this one works well, we have seen it
        # xp = np.clip(x+self.targeted_uaps[y[0]]+self.untargeted_uaps[y[0]], x_min, x_max) # this one works good we have seen it
        # xp = np.clip(x+self.targeted_uaps[y[0]], x_min, x_max)
        perturbed_activations = get_outputs(fm, prepare_dataloader_from_numpy(xp, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        all_activations.append(perturbed_activations)
        
        xu = self.image_processor.local_random_skew(x, strength=0.01)
        # perturbed_u_activations = get_outputs(fm, prepare_dataloader_from_numpy(xu, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        # all_activations.append(perturbed_u_activations)
        
        x_processed_invisible = self.image_processor.process_invisible_perturbations(x, thresh_in=self.e_t)
        processed_activations_inv = get_outputs(fm, prepare_dataloader_from_numpy(x_processed_invisible, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        all_activations.append(processed_activations_inv)
        
        x_processed_edged, x_processed_edged_non_recreated = self.image_processor.process_horizontal(xu, thresh_in=self.e_t, recreate=True, get_non_recreated=True)
        processed_activations = get_outputs(fm, prepare_dataloader_from_numpy(x_processed_edged, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        all_activations.append(processed_activations)
        processed_activations_non_recreated = get_outputs(fm, prepare_dataloader_from_numpy(x_processed_edged_non_recreated, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        all_activations.append(processed_activations_non_recreated)
        
        output_dict = {
            l: np.stack([activations[l]-transformed_activations[l] for transformed_activations in all_activations], axis=1)
            # l: np.stack([normalize(activations[l])-normalize(transformed_activations[l], normalization_standard=activations[l]) for transformed_activations in all_activations], axis=1)
            for l in self.layer_numbers
        }
        
        return dict_to_flattened_np_array(output_dict)
    
    
    def analyze(self, x, y, local_verbose: bool=False, **kwargs):
        
        # predicted_y = get_outputs(self.model.model, prepare_dataloader_from_numpy(x, y, batch_size=self.model.model_configuration['batch_size']), return_numpy=True)
        predicted_classes = y # np.argmax(predicted_y, axis=1)
        
        pcs = np.zeros((len(x), 2))
        scores = np.zeros((len(x)))
        all_labels = np.zeros((len(x)))
        all_labels_2 = np.zeros((len(x)))
        for d, dcdr in enumerate(self.deciders):
            if dcdr.target_class in predicted_classes:
                this_dict = self.get_adversarial_penultimate_features(x[y==dcdr.target_class], y[y==dcdr.target_class])
                pcx, score_x, label, label2 = dcdr.analyze(this_dict)
                all_labels[predicted_classes==dcdr.target_class] = label
                pcs[predicted_classes==dcdr.target_class] = pcx
                scores[predicted_classes==dcdr.target_class] = score_x
                all_labels_2[predicted_classes==dcdr.target_class] = label2
                if local_verbose:
                    print(colored(f'Adaptive threshold is {dcdr.adaptive_thresh} for class {dcdr.target_class} because ratio is {dcdr.ratio}', 'green'))
        # all_labels_2 = np.stack(all_labels_2, axis=1) if len(all_labels_2)>1 else np.expand_dims(all_labels_2, axis=1)
        
        return pcs, scores, all_labels, all_labels_2
    
    
    def forward(self, x, y_out=None, detailed: bool=False, local_verbose: bool=False, **kwargs):
        
        y_out_original = y_out if y_out is not None else get_outputs(self.model_for_backup.model, prepare_dataloader_from_numpy(x, np.zeros((len(x))), batch_size=self.model_for_backup.model_configuration['batch_size']), return_numpy=True)
        y_out = y_out_original.copy()
        y_pred = np.argmax(y_out, axis=1)
        
        xk = self.image_processor.local_random_skew(x, strength=0.01)
        y_out_perturbed = get_outputs(self.model_for_backup.model, prepare_dataloader_from_numpy(xk, np.zeros((len(x))), batch_size=self.model_for_backup.model_configuration['batch_size']), return_numpy=True)
        y_pred_perturbed = np.argmax(y_out_perturbed, axis=1)
        # y_out_perturbed = y_out.copy()
        
        pcs, scores, labels, probabilities = self.analyze(x, y_pred, local_verbose=local_verbose)
        
        y_mins = np.min(y_out, axis=1)
        y_second_best = y_out.copy()
        y_second_best[np.arange(len(y_pred)), y_pred] = y_mins
        
        interesting_indices = (y_pred != y_pred_perturbed) & np.array([y_ in self.target_classes for y_ in y_pred])
        vulnerable_indices = (labels==self.bad_label)
        y_out[interesting_indices, :] = y_second_best[interesting_indices]
        y_out[vulnerable_indices, :] = y_second_best[vulnerable_indices]
        
        if detailed:
            return pcs, scores, labels, probabilities, y_out_original if y_out_original is not None else y_out, y_out
        
        return y_out
    
    
    def __forward(self, x, y_out):
        
        y = np.argmax(y_out, axis=1)
        pc, score, label, label_ellipse = self.analyze(x, y)
        
        vulnerable_ind = (y==self.target_class) & (label==self.bad_label) & (label_ellipse>self.cluster_threshold)
        
        # label correction by outputing second most probable label
        y_mins = np.min(y_out, axis=1)
        y_second_best = y_out.copy()
        y_second_best[:, self.target_class] = y_mins
        
        y_out[vulnerable_ind] = y_second_best[vulnerable_ind]
        
        return y_out
    
    
    