import torch
import numpy as np
import math
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
from scipy.stats import t as t_test, f as ellipse_test

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
            label = np.min(np.stack([label, label_2], axis=1).reshape(-1, 2), axis=1)
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
            print(f'Tangent not fit. Only using ellipse.')
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
    
    
class PCA_Analyzer:
    
    def __init__(
        self, 
        model: Torch_Model, fm,
        train_x=None, train_y=None,
        reference_loader=None,
        target_class: int=0,
        epsilon: float=0.5, std: float=0.,
        e_t: float=0.1,
        verbose: bool=False,
        visible_spuriousity_only: bool=False,
        **kwargs
    ):
        
        # super().__init__()
        
        self.e_t = e_t
        
        self.model_for_backup = model
        self.model = Torch_Model(model.data, model.model_configuration, path=model.path)
        self.model.model.load_state_dict(model.model.state_dict())
        
        self.num_classes = model.data.num_classes
        self.target_class = target_class
        
        self.batch_size = self.model.model_configuration['batch_size']
        self.verbose = verbose
        
        # Setting some attack-related variables
        self.attacker = FGSM(self.model)
        self.epsilon = epsilon
        self.std = std
        
        self.clusterer = KMeans(n_clusters=2, n_init='auto', init='k-means++')
        
        self.ref_x, self.ref_y = get_data_samples_from_loader(reference_loader, return_numpy=True)
        self.ref_x = self.ref_x[self.ref_y==target_class]; self.ref_y = self.ref_y[self.ref_y==target_class]
        random_indices = np.random.choice(len(self.ref_x), size=min(len(self.ref_x), 50), replace=False)
        self.ref_x = self.ref_x[random_indices]; self.ref_y = self.ref_y[random_indices]
        
        uap_attack = Universal_Adversarial_Perturbation(self.model)
        uap_x_untargeted = uap_attack.attack(self.ref_x, self.target_class*np.ones_like(self.ref_y), epsilon=0.1, iterations=10, targeted=False)
        uap_x_targeted = uap_attack.attack(self.ref_x, self.target_class*np.ones_like(self.ref_y), epsilon=0.1, iterations=10, targeted=True)
        self.uap_perturbation_untargeted = np.mean(uap_x_untargeted-self.ref_x, axis=0, keepdims=True)
        self.uap_perturbation_targeted = np.mean(uap_x_targeted-self.ref_x, axis=0, keepdims=True)
        
        # Preparing the feature extractor to extract the features of the the last 15 layers
        len_of_layers = len(self.model.get_children())
        num_layers = min((len_of_layers//2)+1, 15) # if not visible_spuriousity_only else 10)
        self.layer_numbers = np.arange(-num_layers, 0, 1)
        # print(colored(f'[PCA Analyzer] layer_numbers: {self.layer_numbers}', 'red'))
        self.feature_model = Dependable_Feature_Activations(self.model, layer_numbers=self.layer_numbers, get_weighted_activations_for_the_last_layer=True, output_type='dict')
        self.feature_model.model.eval()
        
        # Classical image processor for processing images using classical image processing techniques
        self.image_processor = Image_PreProcessor(batch_size=None) #self.model.model_configuration['batch_size'])
        
        self.custom_clusterer = My_Clusterer(hardness=2, std_multiplier=3 if len(self.ref_x)>=30 else 4)
        self.cluster_threshold = 0.5
        self.pca_f_stats = ASNPCA_Stats(use_median=True)
        
        self.replace = False
        self.disable_kmeans_cluster = False
        self.good_label = 0; self.bad_label = 1
        self.poisoning_score = 1
        self.visible_spuriousity_only = visible_spuriousity_only
        
        self.ac_ref_dict, self.ac_ref = self.get_pc_features(self.ref_x, self.ref_y, use_for_referencing=True)
        print(colored(f'[PCA_Analyzer] Reference activations collected from {self.ac_ref.shape} samples.', "red"))
        
        # # random_ref_placeholders = np.random.uniform(0.2, 0.8, size=(100, 3))
        size_of_random_ref = 1000
        random_ref = np.random.choice(np.arange(0.51, 0.81, 0.01), size=size_of_random_ref*self.ac_ref.shape[1], replace=True).reshape(-1, self.ac_ref.shape[1])
        random_ref = random_ref * np.random.choice([-1, 1], size=size_of_random_ref, replace=True).reshape(-1, 1)
        self.pca_sklearn = PCA_of_SKLEARN(np.concatenate([self.ac_ref, random_ref], axis=0), n_components=2, mean_centric=False)
        self.pcs_ref = self.pca_sklearn.transform(self.ac_ref)
        # Computing the weight of each reference sample based on its nearness to other reference samples (only when large number of reference samples are available)
        weight_of_each = []
        for i, value_set in enumerate(self.pcs_ref):
            weight_ = np.sum(np.square(value_set.reshape(1, -1) - self.pcs_ref), axis=1)
            weight_ = np.mean(np.sort(weight_)[:3])
            weight_of_each.append(weight_)
        weight_of_each = np.array(weight_of_each)
        # weight_of_each = np.mean(np.abs(np.expand_dims(values_ref, axis=0) - np.expand_dims(values_ref, axis=1)), axis=(1,2))
        weight_of_each = exponential_normalize(-100*weight_of_each).reshape(-1, 1)
        self.pcs_ref_median = np.sum(weight_of_each*self.pcs_ref, axis=0, keepdims=True)
        
        self.scores_ = np.mean((self.pcs_ref-self.pcs_ref_median)**2, axis=1)
        self.prepare_clusterers()
        
        
        if not self.visible_spuriousity_only:
            self.use_training_data_for_references = True
            self.use_training_data_for_references = self.use_training_data_for_references and (train_x is not None)
            self.use_training_data_for_references = self.use_training_data_for_references and (train_y is not None)
            if self.use_training_data_for_references:
                self.train_x, self.train_y = train_x[train_y==self.target_class], train_y[train_y==target_class]
                if len(self.train_x) > 30:
                    tpa, _, tpa_label, _ = self.analyze(self.train_x, self.train_y)
                    self.custom_clusterer.fit_sklearn(tpa)
                    print(colored(f'[PCA_Analyzer] Updated the sklearn clusterer with the training data', 'light_red'))
                    
                    
            # OOD Analysis for learning tangent
            ood_data = GTSRB(preferred_size=self.model.data.preferred_size, data_means=self.model.data.data_means, data_stds=self.model.data.data_stds)
            if 'mnist' in self.model.model_configuration['dataset_name']:
                ood_data = Channel1_Torch_Dataset(ood_data)
            ood_data = Custom_Dataset(ood_data, train_size=None, max_target=self.model.data.num_classes-1)
            ood_x, ood_y = get_data_samples_from_loader(torch.utils.data.DataLoader(ood_data.train, batch_size=self.model.model_configuration['batch_size'], shuffle=True), return_numpy=True, size=100)
            # ood_ref, _ = self.get_pc_features(ood_x, ood_y)
            at = PGD(self.model, verbose=verbose)
            adv_ood_x = at.attack(ood_x, self.target_class*np.ones_like(ood_y), epsilon=0.05, iterations=10); 
            adv_ood_y = self.target_class*np.ones_like(ood_y)
            
            opa, _, opa_label, _ = self.analyze(ood_x, ood_y, use_tangent=False)
            adv_opa, _, adv_opa_label, _ = self.analyze(adv_ood_x, adv_ood_y, use_tangent=False)
            condition = False
            condition = condition or (self.bad_label in opa_label)
            condition = condition or (self.bad_label in adv_opa_label)
            if condition:
                mean_opa = np.mean(adv_opa[adv_opa_label==self.bad_label], axis=0)
                if (self.bad_label in opa_label): mean_opa = 0.5*np.mean(opa[opa_label==self.bad_label], axis=0) + 0.5*mean_opa
                self.custom_clusterer.fit_tangent(mean_opa)
            self.adv_opa = adv_opa.copy()
            # opa, _, opa_label, _ = self.analyze(ood_x, ood_y, use_tangent=False)
            # condition = False
            # condition = condition or (self.bad_label in opa_label)
            # if condition:
            #     mean_opa = np.mean(opa[opa_label==self.bad_label], axis=0)
            #     self.custom_clusterer.fit_tangent(mean_opa)
            self.adv_opa = opa.copy()
            self.opa = opa.copy()
        
        return
    
    
    def np_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    
    def get_adversarial_penultimate_features(self, x, y, feature_model: Dependable_Feature_Activations=None, get_outlier_indicators: bool=False):
        
        activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        
        all_activations = []
        # xp = self.attacker.attack(x, self.target_class*np.ones_like(y), epsilon=self.epsilon, targeted=True, verbose=self.verbose)
        xp = x + self.uap_perturbation_targeted
        perturbed_activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(xp, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        all_activations.append(perturbed_activations)
        
        # x = x + self.uap_perturbation
        # perturbed_u_activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(xu, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        # all_activations.append(perturbed_u_activations)
        
        # x_processed_wrap = self.image_processor.local_random_skew(x)
        # processed_wrap_act = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x_processed_wrap, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        # all_activations.append(processed_wrap_act)
        
        if (not self.visible_spuriousity_only) or (False):
            x_processed_invisible = self.image_processor.process_invisible_perturbations(x, thresh_in=self.e_t)
            processed_activations_inv = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x_processed_invisible, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
            all_activations.append(processed_activations_inv)
            
            x_processed_edged, x_processed_edged_non_recreated = self.image_processor.process_horizontal(x, thresh_in=0.05, recreate=True, get_non_recreated=True)
            processed_activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x_processed_edged, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
            all_activations.append(processed_activations)
            processed_activations_non_recreated = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x_processed_edged_non_recreated, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
            all_activations.append(processed_activations_non_recreated)
        
        output_dict = {
            l: np.stack([activations[l]-transformed_activations[l] for transformed_activations in all_activations], axis=1)
            # l: np.stack([normalize(activations[l])-normalize(transformed_activations[l], normalization_standard=activations[l]) for transformed_activations in all_activations], axis=1)
            for l in self.layer_numbers
        }
        
        return output_dict
    
    
    def get_pc_features(self, x, y, use_for_referencing: bool=False):
        
        def dict_to_flattened_np_array(features_dict: dict, axis: int=1):
            return np.concatenate([features_dict[layer_num] for layer_num in self.layer_numbers], axis=axis)
        
        def get_norm(features_, ref_features_):
            def norm(in_):
                # return np.linalg.norm(in_-np.mean(ref_features_, axis=0, keepdims=True), axis=1)
                return np.mean(np.square(in_-np.mean(ref_features_, axis=0, keepdims=True)), axis=1)
            actual_norm = norm(features_)
            ref_norm = norm(ref_features_)
            return normalize(actual_norm, normalization_standard=ref_norm)
        
        def get_feature_from_dict_hypothesis(features_dict: dict, ref_dict: dict=None):
            ref_dict = self.ac_ref_dict if ref_dict is None else ref_dict
            flattened_ref = dict_to_flattened_np_array(ref_dict, axis=2)
            flattened_dict = dict_to_flattened_np_array(features_dict, axis=2)
            outputs, outputs_mag, outputs_2 = [], [], []
            for i in range(flattened_ref.shape[1]):
                outputs.append(cosine_similarity(flattened_dict[:,i], np.median(flattened_ref[:,i], axis=0, keepdims=True)))
                if i != 0:
                    value = cosine_similarity(flattened_dict[:,i], flattened_dict[:,0])
                    outputs_2.append(value[np.arange(len(value)), np.arange(len(value))])
                # mag = 1/(1+get_norm(flattened_dict[:, i], flattened_ref[:, i]))
                # outputs_mag.append(2*mag-1)
            outputs += [np.stack(outputs_2, axis=1)] if len(outputs_2)>0 else []
            outputs += [np.stack(outputs_mag, axis=1)] if len(outputs_mag)>0 else []
            return np.concatenate(outputs, axis=1) if len(outputs)>1 else np.concatenate(outputs+outputs, axis=1)
        
        def get_feature_from_dict_hypothesis_with_magnitude(features_dict: dict, ref_dict: dict=None):
            ref_dict = self.ac_ref_dict if ref_dict is None else ref_dict
            flattened_ref = dict_to_flattened_np_array(ref_dict, axis=2)
            flattened_dict = dict_to_flattened_np_array(features_dict, axis=2)
            outputs, outputs_2, outputs_3 = [], [], []
            for i in range(flattened_ref.shape[1]):
                outputs.append(cosine_similarity(flattened_dict[:,i], np.mean(flattened_ref[:,i], axis=0, keepdims=True)))
                if i == 0:
                    outputs_3.append(np.linalg.norm(flattened_dict[:,i], axis=1))
                if i != 0:
                    value = cosine_similarity(flattened_dict[:,i], flattened_dict[:,0])
                    outputs_2.append(value[np.arange(len(value)), np.arange(len(value))])
            outputs += [np.stack(outputs_2, axis=1), np.stack(outputs_3, axis=1)]
            return np.concatenate(outputs, axis=1)
        
        def get_feature_from_dict(features_dict: dict, ref_dict: dict=None):
            ref_dict = self.ac_ref_dict if ref_dict is None else ref_dict
            all_keys = self.layer_numbers
            
            res_1 = {
                key: 
                    np.concatenate([
                        cosine_similarity(features_dict[key][:,l], np.mean(ref_dict[key][:,l], axis=0, keepdims=True))
                        # cosine_similarity(features_dict[key][:,l], features_dict[key][:,0])
                        for l in range(len(features_dict[key][0]))
                    ], axis=1)
                for key in all_keys
            }
            
            res_2 = {}
            for key in all_keys:
                feature_ = features_dict[key]
                values = []
                for l in range(1, len(feature_[0])):
                    value = cosine_similarity(feature_[:,l], feature_[:,0])
                    value = value[np.arange(len(value)), np.arange(len(value))]
                    values.append(value)
                values = np.stack(values, axis=1)
                res_2[key] = values
            
            return np.concatenate([dict_to_flattened_np_array(res_1), dict_to_flattened_np_array(res_2)], axis=1)
        
        _p_activations = self.get_adversarial_penultimate_features(x, y)
        activations = get_feature_from_dict_hypothesis(_p_activations, ref_dict=_p_activations if use_for_referencing else None)
        # activations = get_feature_from_dict_hypothesis_with_magnitude(_p_activations, ref_dict=_p_activations if use_for_referencing else None)
        
        if use_for_referencing:
            return deepcopy(_p_activations), activations.copy()
        
        return activations, activations.copy()
    
    
    def prepare_clusterers(self):
        
        # =============
        # Ellipse Cluster things
        # =============
        self.limited_pa = self.pcs_ref-self.pcs_ref_median
        self.custom_clusterer.fit(self.limited_pa)
        # # print(f'\nGoing to analyze for the ellipse cluster.')
        # ellipse_iterations = 1
        # use_reference_samples = True
        # limited_x, limited_y = self.ref_x, self.ref_y
        
        # limited_pa = []
        # for i in range(ellipse_iterations):
        #     _limited_pa, _limited_scores, _, _ = self.analyze(limited_x, limited_y)
        #     limited_pa.append(_limited_pa)
        # self.limited_pa_ = np.concatenate(limited_pa, axis=0)
        # self.limited_pa = np.median(limited_pa, axis=0)
        # # print(f'Analyzed dataset for the ellipse cluster.')
        # self.custom_clusterer.fit(self.limited_pa_)
        
        
        # # =============
        # # K-Means Cluster things
        # # =============
        # metric_scores = deepcopy(self.pcs_[:, -2:])
        # self.metric_scores = np.append(metric_scores, self.scores_.reshape(-1, 1), axis=1)
        # self.clusterer.fit(self.metric_scores)
        
        # labels_ref = self.clusterer.predict(np.append(_limited_pa, _limited_scores.reshape(-1, 1), axis=1))
        # label_scores = [np.mean(labels_ref), 1-np.mean(labels_ref)]
        # self.good_label = np.argmin(label_scores); self.bad_label = np.argmax(label_scores)
        # if np.max(label_scores)<0.85:
        #     print(f'Disabling kmeans cluster.')
        #     self.disable_kmeans_cluster = True
        
        # # =============
        # # Compute the poisoning score of the classifier
        # # =============
        # normalizing_max = np.max(self.pca_stats.pc_dict['max'])
        # normalizing_min = np.min(self.pca_stats.pc_dict['min'])
        # self.poisoning_score = np.abs(normalizing_max + normalizing_min)
        
        # print(colored('\n***********************************************', 'light_green'))
        # print(colored(f'***** Poisoning Score: {self.poisoning_score:.5f} ***********', 'light_green'))
        # print(colored('***********************************************', 'light_green'))
        
        return
    
    
    def analyze(self, x, y, cluster_xy: bool=True, use_tangent: bool=True, **kwargs):
        
        activations, p_activations = self.get_pc_features(x, y)
        
        pcs_s = self.pca_sklearn.transform(activations) - self.pcs_ref_median
        scores_s = np.mean(pcs_s**2, axis=1)
        # scores_s = self.mahalanobis_distance(pcs_s)
        
        label = self.good_label*np.ones((len(x)))
        _label = np.zeros_like(label)
        if self.custom_clusterer.configured and cluster_xy:
            _label = self.custom_clusterer.predict(pcs_s[:, 0], pcs_s[:, 1], use_tangent=use_tangent)
            # label[_label<0.5] = self.good_label
            
            label = _label.copy()
            label[_label<=self.cluster_threshold] = self.good_label
            label[_label>self.cluster_threshold] = self.bad_label
        else:
            print(colored(f'Not using clusterer because it is not configured.', 'light_red'))
        
        return pcs_s, scores_s, label, _label # if self.custom_clusterer.configured else label
    
    
    def forward(self, x, y_out):
        
        y = np.argmax(y_out, axis=1)
        pc, score, label, label_ellipse = self.analyze(x, y)
        
        vulnerable_ind = (y==self.target_class) & (label==self.bad_label) & (label_ellipse>self.cluster_threshold)
        # vulnerable_ind_for_processed = (y==self.target_class) & (label==self.good_label) & (label_ellipse>self.cluster_threshold)
        
        # # processed label correction
        # y_processed = get_outputs(self.model.model, prepare_dataloader_from_numpy(self.image_processor.process_horizontal(x), x, batch_size=self.batch_size), return_numpy=True)
        
        # label correction by outputing second most probable label
        y_mins = np.min(y_out, axis=1)
        y_second_best = y_out.copy()
        y_second_best[:, self.target_class] = y_mins
        
        # # label correction by outputing random label
        # y_random = np.random.normal(0, self.num_classes, size=y_out.shape)
        # y_random[y_random==self.target_class] += 1
        # y_random = y_random % self.num_classes
        
        y_out[vulnerable_ind] = y_second_best[vulnerable_ind]
        # y_out[vulnerable_ind_for_processed] = y_processed[vulnerable_ind_for_processed]
        
        return y_out
    
    
    def process_general(self, x: np.ndarray, *args, **kwargs):
        out = self.image_processor.process_horizontal(x)
        out = self.process_old(x)
        return out
    
    
    def process_old(self, x: np.ndarray, smoothing_iterations: int=4):
        
        thresh = 0.05
        def torch_convolution_cuda(image: torch.tensor, kernel: torch.tensor):
            return torch.nn.functional.conv2d(image.to('cuda'), kernel.to('cuda'), stride=1, padding='same').detach().cpu()

        def get_edges(image: np.ndarray):
            filter = 0.25 * np.array(
                [np.array([0,1,0]),
                np.array([1,-4,1]),
                np.array([0,1,0])]
            )
            filter = np.expand_dims(filter, axis=0)
            filter = np.expand_dims(filter, axis=0).astype(np.float32)
            output = torch_convolution_cuda(torch.tensor(np.mean(image, axis=1, keepdims=True)), torch.tensor(filter)).numpy()
            output[output < thresh] = 0.
            output[output >= thresh] = 1.
            return output

        def smooth(image: np.ndarray):
            filter = (1/9) * np.array(
                [
                    np.array([1,1,1]),
                    np.array([1,1,1]),
                    np.array([1,1,1]),
                    
                ]
            )
            zeros = np.zeros_like(filter)
            filter_3 = np.stack((zeros, zeros, filter), axis=0)
            filter_2 = np.stack((zeros, filter, zeros), axis=0)
            filter_1 = np.stack((filter, zeros, zeros), axis=0)
            filter = np.stack([filter_1, filter_2, filter_3], axis=0).astype(np.float32)
            output = image.copy()
            for i in range(1):
                output = torch_convolution_cuda(torch.tensor(output), torch.tensor(filter)).numpy()
            return output

        def process_image(image: np.ndarray):
            output_smooth = image.copy()
            output_edges = get_edges(output_smooth)
            output_edges = np.concatenate([output_edges, output_edges, output_edges], axis=1)
            
            # set random pixels in output edges to 1 in order to maintain color marks
            output_edges2 = np.zeros_like(output_edges[:, 0])
            output_edges2[np.random.rand(*output_edges2.shape) < 0.05] = 1
            output_edges2 = np.stack([output_edges2, output_edges2, output_edges2], axis=1)
            output_edges2[output_edges==1] = 0
            
            for i in range(smoothing_iterations):
                output_smooth = smooth(output_smooth)
                output_smooth[output_edges==1] = image[output_edges==1]
                output_smooth[output_edges2==1] = image[output_edges2==1]
            output_smooth_further = smooth(output_smooth.copy())
            output_smooth[output_edges2==1] = output_smooth_further[output_edges2==1]
            
            return output_smooth
        
        return process_image(x)
    
    
    def mahalanobis_distance(self, point, eps=1e-8):
        """
        Compute the Mahalanobis distance of a 2D point from a distribution of N 2D samples.

        Parameters
        ----------
        point : array-like, shape (2,) or (M, 2)
            The query point(s).
        samples : array-like, shape (N, 2)
            Distribution of 2D sample points.
        eps : float
            Small regularization term for stable inverse covariance.

        Returns
        -------
        d : float or np.ndarray
            Mahalanobis distance(s). If `point` is (2,), returns scalar.
            If (M,2), returns array of length M.
        """
        
        samples = np.asarray(self.pcs_ref-self.pcs_ref_median)
        point = np.asarray(point)

        # Compute mean and covariance of the distribution
        mean = samples.mean(axis=0)               # (2,)
        cov = np.cov(samples, rowvar=False)       # (2,2)

        # Regularize covariance to avoid singular inversion
        cov += eps * np.eye(2)

        # Precompute inverse covariance
        cov_inv = np.linalg.inv(cov)

        # Center points
        diff = point - mean   # (2,) or (M,2)

        # Mahalanobis formula:
        # d = sqrt( (x - μ)^T Σ^{-1} (x - μ) )
        if diff.ndim == 1:
            return np.sqrt(diff.T @ cov_inv @ diff)
        else:
            return np.sqrt(np.einsum('ij, jk, ik -> i', diff, cov_inv, diff))
        
        