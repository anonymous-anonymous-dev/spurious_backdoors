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
            # if self.reference_activations is not None:
            #     pc_r = pca_analyzer.transform(self.reference_activations[0]).reshape(len(self.reference_activations[0]), -1)
            #     pc_rp = pca_analyzer.transform(self.reference_activations[1]).reshape(len(self.reference_activations[1]), -1)
            #     pc -= np.median(pc_r, axis=0, keepdims=True)
            #     pc_p -= np.median(pc_rp, axis=0, keepdims=True)
            pc = self.normalize(pc, self.pc_dict)
            pc_p = self.normalize(pc_p, self.pc_p_dict)
            self.pca_analyzers.append(pca_analyzer)
        else:
            pc = self.pca_analyzers[i].transform(activations).reshape(len(activations), -1)
            pc_p = self.pca_analyzers[i].transform(p_activations).reshape(len(p_activations), -1)
            # if self.reference_activations is not None:
            #     pc_r = self.pca_analyzers[i].transform(self.reference_activations[0]).reshape(len(self.reference_activations[0]), -1)
            #     pc_rp = self.pca_analyzers[i].transform(self.reference_activations[1]).reshape(len(self.reference_activations[1]), -1)
            #     pc -= np.median(pc_r, axis=0, keepdims=True)
            #     pc_p -= np.median(pc_rp, axis=0, keepdims=True)
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
    
    
    
class PCA_Analyzer:
    
    def __init__(
        self, 
        model: Torch_Model, feature_model: Feature_Activations, 
        x, y, ac_, 
        epsilon=0.5,
        mode='one', sample_subset: int=30, iterations: int=100, std: float=0.2,
        threshold: float=0.35,
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
        
        self.pca_stats = ASNPCA_Stats(use_median=True)
        self.attacker = FGSM(self.model)
        self.epsilon = epsilon
        self.std = std
        
        self.clusterer = KMeans(n_clusters=2, n_init='auto', init='k-means++')
        # self.clusterer = GaussianMixture(n_components=2)
        
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
        
        thresh = 0.1
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
                [np.array([1,1,1]),
                np.array([1,1,1]),
                np.array([1,1,1])]
            )
            zeros = np.zeros_like(filter)
            filter_3 = np.stack((zeros, zeros, filter), axis=0)
            filter_2 = np.stack((zeros, filter, zeros), axis=0)
            filter_1 = np.stack((filter, zeros, zeros), axis=0)
            filter = np.stack([filter_1, filter_2, filter_3], axis=0).astype(np.float32)
            output = image.copy()
            for i in range(2):
                output = torch_convolution_cuda(torch.tensor(output), torch.tensor(filter)).numpy()
            return output
        def process(image: np.ndarray):
            output_edges = get_edges(image)
            output_edges = np.concatenate([output_edges, output_edges, output_edges], axis=1)
            output_smooth = smooth(image)
            output_smooth[output_edges==1] = image[output_edges==1]
            return output_smooth
        
        # show_image_grid(normalize(x), n_rows=1, n_cols=5, channels_first=True);
        # x = process(x)
        # show_image_grid(normalize(x), n_rows=1, n_cols=5, channels_first=True);
        
        noise = np.random.normal(0, self.std*np.std(x), size=x.shape).astype(np.float32)
        x = x + noise
        xp = self.attacker.attack(x, self.target_class*np.ones_like(y), epsilon=self.epsilon, targeted=True, verbose=self.verbose)
        
        # _mnpca._model.mode = 'only_activations'
        activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        p_activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(xp, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        # _mnpca._model.mode = 'default'
            
        return activations, p_activations
    
    
    def prepare_analyzer_smooth(self, x, y):
        
        activations, p_activations = self.get_penulatimate_features(x, y)
        
        # pcs_s = []; scores_s = []
        for i in range(self.iterations):
            pca_analyzer = PCA_of_SKLEARN(self.ac_[np.random.choice(len(self.ac_), size=self.sample_subset, replace=self.replace)], n_components=1)
            pcs, scores = self.pca_stats.update_stats(pca_analyzer, activations, p_activations)
            # pcs_s.append(pcs); scores_s.append(scores)
            
        # new thing
        pcs_s = []; scores_s = []
        for i in range(self.iterations):
            pcs, scores = self.pca_stats.compute_pcs_and_score(activations, p_activations, i)
            pcs_s.append(pcs); scores_s.append(scores)
        self.pcs_ = np.mean(pcs_s, axis=0)
        self.scores_ = np.mean(scores_s, axis=0)
        
        # =============
        # Cluster things
        # =============
        metric_scores = deepcopy(self.pcs_[:, -2:])
        self.metric_scores = np.append(metric_scores, self.scores_.reshape(-1, 1), axis=1)
        # self.poisoning_score = 0.2; self.good_label=0; self.bad_label=1
        self.clusterer.fit(self.metric_scores)
        labels = self.clusterer.predict(self.metric_scores)
        
        # good_condition = (np.abs(self.pcs_[:, 1])<0.2)
        # self.good_label = 0 if np.mean(labels[good_condition])<0.5 else 1
        # self.bad_label = 1 - self.good_label
        label_lengths = [np.mean(labels==0), np.mean(labels!=0)]
        self.good_label = np.argmax(label_lengths); self.bad_label = np.argmin(label_lengths)
        good_l = (labels==self.good_label)
        bad_l = (labels==self.bad_label)
        max_l = np.mean(good_l)
        self.poisoning_score_self = np.mean(self.pcs_[bad_l, 1])*max_l - np.mean(self.pcs_[good_l, 1])*(1-max_l)
        
        # pcs_0_greater = np.clip(np.abs(self.pcs_[:,0])-0.2, 0, 1)
        # pcs_1_greater = np.clip(np.abs(self.pcs_[:,1])-0.15, 0, 1)
        # # self.poisoning_score = np.mean( pcs_0_greater * pcs_1_greater ) * np.max(pcs_1_greater)
        
        # sorted_pcs = np.sort(np.abs(self.pcs_[:, 1]))
        # self.poisoning_score = np.median(sorted_pcs[-(1+len(sorted_pcs)//100):])
        
        normalizing_max = np.max(self.pca_stats.pc_p_dict['max'])
        normalizing_min = np.min(self.pca_stats.pc_p_dict['min'])
        # normalizing_stds = np.std(self.pca_stats.pc_p_dict['values'])
        # normalizing_values_ = np.max(self.pca_stats.pc_dict['max'])
        # normalizing_min_ = np.min(self.pca_stats.pc_dict['min'])
        # normalizing_stds_ = np.std(self.pca_stats.pc_dict['values'])
        self.poisoning_score = np.abs(normalizing_max + normalizing_min)
        # self.poisoning_score = normalizing_max - normalizing_min
        
        sorted_pcs = np.sort(self.pcs_[:, 1])
        ind_point_1 = max(len(sorted_pcs) // 100, 1)
        min_median = np.mean(sorted_pcs[:ind_point_1]); max_median = np.mean(sorted_pcs[-ind_point_1:])
        _range = max(-min_median, max_median)
        
        
        print(colored('\n***********************************************', 'light_green'))
        print(colored(f'***** Poisoning Score: {self.poisoning_score:.5f} ***********', 'light_green'))
        print(colored(f'***** Poisoning Score self: {self.poisoning_score_self:.5f} ***********', 'light_green'))
        print(colored(f'***** Poisoning Score potential: {normalizing_max**2+normalizing_min**2:.5f} ***********', 'light_green'))
        print(colored(f'***** Range: {_range:.5f} ***********', 'light_green'))
        print(colored('***********************************************', 'light_green'))
        
        self.liar = 1 if -min_median<max_median else -1
        # # self.poisoning_score = 0.2
        
        
        gm = GaussianMixture(n_components=2, random_state=0).fit(self.pcs_)
        print(gm.means_)
        print(colored('***********************************************', 'light_green'))
        
        
        return
    
    
    def pairwise_cosine_similarity_torch(self, flattened_clients_states):
        flattened_clients_states = torch.tensor(flattened_clients_states)
        normalized_input_a = torch.nn.functional.normalize(flattened_clients_states)
        res = torch.mm(normalized_input_a, normalized_input_a.T)
        res[res==0] = 1e-6
        return res.detach().cpu().numpy()
    
    
    def analyze(self, x, y):
        
        activations, p_activations = self.get_penulatimate_features(x, y)
        
        pcs_s = []; scores_s = []
        for i in range(self.iterations):
            pcs, scores = self.pca_stats.compute_pcs_and_score(activations, p_activations, i)
            pcs_s.append(pcs); scores_s.append(scores)
            
        pcs_s = np.mean(pcs_s, axis=0)
        scores_s = np.mean(scores_s, axis=0)
        metric_scores = deepcopy(pcs_s[:, -2:])
        metric_scores = np.append(metric_scores, scores_s.reshape(-1, 1), axis=1)
        
        label = self.clusterer.predict(metric_scores)
        if (self.poisoning_score > self.threshold) and (self.scores_ < 0.8):
            label = self.good_label*np.ones_like(label)
            label[self.liar * pcs_s[:,1] < 0.15] = self.good_label
            label[self.liar * pcs_s[:,1] >= 0.15] = self.bad_label
        
        return pcs_s, scores_s, label if self.poisoning_score>self.threshold else self.good_label*np.ones_like(label)
    
    
    def forward(self, x, y_out):
        
        y = np.argmax(y_out, axis=1)
        pc, score, label = self.analyze(x, y)
        
        vulnerable_ind = (y==self.target_class)
        vulnerable_ind = vulnerable_ind & (label==self.bad_label)
        
        y_random = np.random.normal(0, self.num_classes, size=y_out.shape)
        y_random[y_random==self.target_class] += 1
        y_random = y_random % self.num_classes
        y_out[vulnerable_ind] = y_random[vulnerable_ind]
        
        return y_out
    
    
    