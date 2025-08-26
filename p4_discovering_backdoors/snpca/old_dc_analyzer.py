import torch
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from _0_general_ML.model_utils.torch_model import Torch_Model
from _1_adversarial_ML.adversarial_attacks.all_available_adversarial_attacks import FGSM
from .fgsm_torch import FGSM_Torch

from ..model_utils.feature_activations import Feature_Activations

from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy, get_outputs, prepare_dataloader_from_tensor
from utils_.pca import PCA_of_SKLEARN



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
    
    
    
class PCA_Analyzer:
    
    def __init__(
        self, 
        model: Torch_Model, feature_model: Feature_Activations, 
        x, y, ac_, 
        epsilon=0.5,
        mode='multi', sample_subset: int=30, iterations: int=100,
        threshold: float=0.65,
        verbose: bool=False
    ):
        
        # super().__init__()
        
        self.pca_stats = ASNPCA_Stats()
        
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
        
        self.attacker = FGSM(self.model)
        self.epsilon = epsilon
        
        self.clusterer = KMeans(n_clusters=2, n_init='auto', init='random')
        
        self.mode = mode
        if self.mode == 'one':
            self.iterations = 1
            self.sample_subset = len(self.ac_)
            self.replace = False
            self.prepare_analyzer_smooth(x, y)
        else:
            self.iterations = iterations
            self.sample_subset = sample_subset
            self.replace = True
            self.prepare_analyzer_smooth(x, y)
        
        return
    
    
    def np_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    
    def __get_penulatimate_features(self, x, y):
        
        # print(x.shape)
        activations, p_activations = [], []
        for i in range(20):
            print(f'\rSmoothing perturbations: {i}/{100}.', end='')
            
            noise = np.random.normal(0, 0.2*np.std(x), size=x.shape).astype(np.float32)
            x = x + noise
            xp = self.attacker.attack(x, self.target_class*np.ones_like(y), epsilon=self.epsilon, targeted=True, verbose=self.verbose)
            
            # _mnpca._model.mode = 'only_activations'
            _activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
            _p_activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(xp, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
            # _mnpca._model.mode = 'default'
            
            activations.append(_activations)
            p_activations.append(_p_activations)
        
        print()
        activations = np.mean(activations, axis=0)
        p_activations = np.mean(p_activations, axis=0)
            
        return activations, p_activations
    
    
    def get_penulatimate_features(self, x, y):
        
        noise = np.random.normal(0, 0.2*np.std(x), size=x.shape).astype(np.float32)
        x = x + noise
        xp = self.attacker.attack(x, self.target_class*np.ones_like(y), epsilon=self.epsilon, targeted=True, verbose=self.verbose)
        
        # _mnpca._model.mode = 'only_activations'
        activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        p_activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(xp, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        # _mnpca._model.mode = 'default'
            
        return activations, p_activations
    
    
    def prepare_analyzer_smooth(self, x, y):
        
        activations, p_activations = self.get_penulatimate_features(x, y)
        
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
        for i in range(self.iterations):
            pca_analyzer = PCA_of_SKLEARN(self.ac_[np.random.choice(len(self.ac_), size=self.sample_subset, replace=self.replace)], n_components=1)
        
            pc = pca_analyzer.transform(activations).reshape(len(x), -1)
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
            
        self.pcs_ = np.mean(self.pcs_s, axis=0)
        self.scores_ = np.mean(self.scores_s, axis=0)
        metric_scores = np.append(self.pcs_, self.scores_.reshape(-1, 1), axis=1)
        
        metric_2 = self.pcs_ * self.scores_.reshape(-1, 1)
        self.clusterer.fit(metric_scores)
        labels = self.clusterer.labels_
        
        label_lengths = [np.mean(labels==0), np.mean(labels!=0)]
        self.good_label = np.argmax(label_lengths); self.bad_label = np.argmin(label_lengths)
        
        # # strategy 1 for poisoning score
        # self.ss = silhouette_score(metric_2, labels)
        # self.poisoning_score = self.ss*(1-np.min(label_lengths))
        
        # strategy 2 for poisoning score
        good_l = (labels==self.good_label)
        bad_l = (labels==self.bad_label)
        max_l = np.mean(good_l)
        self.poisoning_score = np.mean(self.scores_[bad_l]) - np.mean(self.scores_[good_l]) * max_l
        
        return
    
    
    def analyze(self, x, y):
        
        activations, p_activations = self.get_penulatimate_features(x, y)
        
        pcs_s = []; scores_s = []
        for i in range(self.iterations):
            pc = self.pca_analyzers[i].transform(activations).reshape(len(x), -1)
            pc_p = self.pca_analyzers[i].transform(p_activations).reshape(len(p_activations), -1)

            pc = (pc-self.pc_mins[i])/(self.pc_maxs[i]-self.pc_mins[i]); #pc = pc - np.median(pc)
            pc_p =(pc_p-self.pc_p_mins[i])/(self.pc_p_maxs[i]-self.pc_p_mins[i]); #pc_p = pc_p - np.median(pc_p)
            pcs_ = np.append(pc, pc_p, axis=1)
            pcs_ = pcs_ - self.pc_medians[i]
            
            scores_ = np.mean(pcs_**2, axis=1)
            # scores_ += np.mean((p_activations-self.pca_analyzers[i].reconstruct(p_activations))**2, axis=1)
            scores_ += np.mean((activations-self.pca_analyzers[i].reconstruct(activations))**2, axis=1)
            scores_ = (scores_-self.scores_mins[i])/(self.scores_maxs[i]-self.scores_mins[i])
            
            pcs_s.append(pcs_); scores_s.append(scores_)
            
        pcs_s = np.mean(pcs_s, axis=0)
        scores_s = np.mean(scores_s, axis=0)
        metric_scores = np.append(pcs_s, scores_s.reshape(-1, 1), axis=1)
        label = self.clusterer.predict(metric_scores)
    
        return pcs_s, scores_s, label if self.poisoning_score>self.threshold else self.good_label*np.ones_like(label)
    
    
    def __analyze(self, x, y):
        if self.mode == 'one': return self.analyze_one(x, y)
        return self.analyze_multi(x, y)
        
        
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
    
    
    def __prepare_analyzer_smooth_one(self, x, y):
        
        xp = x+np.random.normal(0, 0.02*np.std(x), size=x.shape).astype(np.float32)
        xp = self.attacker.attack(xp, self.target_class*np.ones_like(y), epsilon=self.epsilon, targeted=True, verbose=self.verbose)

        # _mnpca._model.mode = 'only_activations'
        activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        p_activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(xp, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        # _mnpca.model.mode = 'default'
        
        self.pca_analyzer = PCA_of_SKLEARN(self.ac_, n_components=1)
        
        pc = self.pca_analyzer.transform(activations).reshape(len(x), -1)
        pc_p = self.pca_analyzer.transform(p_activations).reshape(len(xp), -1)
        
        self.pc_max = np.max(pc)
        self.pc_min = np.min(pc)
        self.pc_p_max = np.max(pc_p)
        self.pc_p_min = np.min(pc_p)
        
        pc = (pc-self.pc_min)/(self.pc_max-self.pc_min); #pc = pc - np.median(pc)
        pc_p =(pc_p-self.pc_p_min)/(self.pc_p_max-self.pc_p_min); #pc_p = pc_p - np.median(pc_p)
        pc_ = np.append(pc, pc_p, axis=1)
        
        self.pc_median = np.median(pc_, axis=0, keepdims=True)
        
        self.pcs_ = pc_ - self.pc_median
        scores_ = np.mean(self.pcs_**2, axis=1)
        
        self.scores_max = np.max(scores_)
        self.scores_min = np.min(scores_)
        self.scores_ = (scores_-self.scores_min)/(self.scores_max-self.scores_min)
        
        metric_scores = np.append(self.pcs_, self.scores_.reshape(-1, 1), axis=1)
        metric_2 = self.pcs_ * self.scores_.reshape(-1, 1)
        self.clusterer.fit(metric_scores)
        labels = self.clusterer.labels_
        
        # # strategy 1 for poisoning score
        # ss = silhouette_score(metric_2, labels)
        # label_lengths = [np.mean(labels==0), np.mean(labels!=0)]
        # self.good_label = np.argmax(label_lengths); self.bad_label = np.argmin(label_lengths)
        # self.poisoning_score = ss*(1-np.min(label_lengths))
        
        # strategy 2 for poisoning score
        good_l = (labels==self.good_label)
        bad_l = (labels==self.bad_label)
        max_l = np.mean(good_l)
        self.poisoning_score = np.mean(self.scores_[bad_l]) - np.mean(self.scores_[good_l]) * max_l
        self.threshold = 0.2
        
        return
    
    
    def __analyze_one(self, x, y):
        
        xp = x+np.random.normal(0, 0.02*np.std(x), size=x.shape).astype(np.float32)
        xp = self.attacker.attack(xp, self.target_class*np.ones_like(y), epsilon=self.epsilon, targeted=True, verbose=self.verbose)
        
        # _mnpca._model.mode = 'only_activations'
        activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(x, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        p_activations = get_outputs(self.feature_model, prepare_dataloader_from_numpy(xp, y, batch_size=self.batch_size), return_numpy=True, verbose=self.verbose)
        # _mnpca.model.mode = 'default'
        
        pc = self.pca_analyzer.transform(activations).reshape(len(x), -1)
        pc_p = self.pca_analyzer.transform(p_activations).reshape(len(xp), -1)

        pc = (pc-self.pc_min)/(self.pc_max-self.pc_min); #pc = pc - np.median(pc)
        pc_p =(pc_p-self.pc_p_min)/(self.pc_p_max-self.pc_p_min); #pc_p = pc_p - np.median(pc_p)
        pc_ = np.append(pc, pc_p, axis=1)
        pc_ = pc_ - self.pc_median
        score_ = np.mean(pc_**2, axis=1)
        score_ = (score_-self.scores_min)/(self.scores_max-self.scores_min)
        
        label = self.clusterer.predict(np.append(pc_, score_.reshape(-1, 1), axis=1))
    
        return pc_, score_, label if self.poisoning_score>self.threshold else self.good_label*np.ones_like(label)
    
    
    