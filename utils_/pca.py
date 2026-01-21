import numpy as np
import torch

from sklearn.decomposition import PCA, SparsePCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler


from .general_utils import normalize, de_normalize



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PCA_Loss:
    
    def __init__(self, mean: np.ndarray, components: np.ndarray, target_class: int, mode: str='official', device: str=torch_device):
        
        self.mean = torch.tensor(mean.astype(np.float32), dtype=torch.float32).to(device)
        self.components = torch.tensor(components.astype(np.float32), dtype=torch.float32).to(device)
        self.target_class = target_class
        
        self.mode = mode
        self.device = device
        
        return
    
    
    def transform(self, activations, eigvec_scale=True):
        
        act = activations.float().to(self.device) - self.mean.to(self.device)
        act_pca = act@self.components.to(self.device)
        if eigvec_scale:
            act_pca = act_pca * torch.sum(self.components.to(self.device), dim=0)
        
        return act_pca
    
    
    def __official_call__(self, alpha: torch.tensor, y):
        
        alpha = alpha@self.components
        alpha = -alpha * torch.sum(self.components, dim=0, keepdim=True)
        
        return alpha[np.arange(len(y)), y.long()]
    
        
    def __unofficial_call__(self, alpha: torch.tensor, y):
        
        alpha = alpha / torch.sum(torch.square(alpha), dim=1, keepdim=True)
        components = self.components / torch.sum(torch.square(self.components), dim=1, keepdim=True)
        
        alpha = -alpha@components
        
        return alpha[np.arange(len(y)), y.long()]
    
    
    def __call__(self, alpha, y):
        return self.__official_call__(alpha, y) if self.mode!='unofficial' else self.__unofficial_call__(alpha, y)
    
    
    
class General_PCA:
    
    def __init__(self, X, **kwargs):
        self.variances, self.components, self.mean = self.compute_pca(X, **kwargs)
        return
    
    def not_implemented(self):
        raise NotImplementedError('This function has not been implemented. Please call the child class.')
    
    
    def compute_pca(self, *args, **kwargs): return self.not_implemented()
    def transform(self, *args, **kwargs) -> np.ndarray: return self.not_implemented()
    def inverse_transform(self, *args, **kwargs) -> np.ndarray: return self.not_implemented()
    
    def reconstruct(self, X, remove_indices: int=None) -> np.ndarray:
        transformed_X = self.transform(X)
        if remove_indices is not None:
            transformed_X[:, remove_indices] *= 0.
        return self.inverse_transform(transformed_X)
    
    

class PCA_of_NPCA(General_PCA):
    
    def __init__(self, X: np.ndarray, n_components: int=None, **kwargs):
        
        self.n_components = n_components if n_components is not None else X.shape[1]
        super().__init__(X.T) # NPCA implementation uses the transposed inputs for evaluations.
        
        return
    
    
    def compute_pca(self, X, **kwargs):
        """
        X - DxN np.ndarray, datapoints
        """
        # center data
        X_mean = np.mean(X, axis=1)
        X_centered = X - X_mean[:, np.newaxis]
        
        # covariance matrix
        cov = np.cov(X_centered)
        
        # sort by eigenvalues
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        sort_idcs = np.argsort(eig_vals)[::-1][:self.n_components]
        
        return eig_vals[sort_idcs], eig_vecs[:, sort_idcs], X_mean.T
    
    
    def transform(self, activations, **kwargs):
        
        if isinstance(activations, np.ndarray):
            act_pca = self.transform_numpy(activations, **kwargs)
        else:
            act_pca = self.transform_torch(activations, **kwargs)
        
        return act_pca
    
    
    def transform_numpy(self, activations, eigvec_scale: bool=True, **kwargs):
        act = activations - self.mean
        act_pca = act@self.components
        if eigvec_scale:
            act_pca = act_pca * np.sum(self.components, axis=0)
        return act_pca
    
    
    def transform_torch(self, activations: torch.Tensor, eigvec_scale: bool=True, device='cuda'):
        
        # device = activations.device
        act = activations.to(device) - torch.tensor(self.mean).to(device)
        act_pca = act@torch.tensor(self.components, dtype=torch.float32).to(device)
        if eigvec_scale:
            act_pca = act_pca * torch.sum(torch.tensor(self.components, dtype=torch.float32).to(device), dim=0)
        
        return act_pca

    def inverse_transform_numpy(self, act_pca):
        return (np.matmul(act_pca, self.components.T) + self.mean).astype(np.float32)
    
    
    def inverse_transform_torch(self, act_pca, device='cuda'):
        return act_pca@torch.tensor(self.components, dtype=torch.float32).to(device).T + torch.tensor(self.mean, dtype=torch.float32).to(device)
    
    
    def inverse_transform(self, act_pca, **kwargs):
        
        if isinstance(act_pca, np.ndarray):
            act_pca = self.inverse_transform_numpy(act_pca, **kwargs)
        else:
            act_pca = self.inverse_transform_torch(act_pca, **kwargs)
        
        return act_pca



class PCA_of_SKLEARN(General_PCA):
    
    def __init__(self, X: np.ndarray, n_components=None, normalization: bool=False, standardization: bool=False, mean_centric: bool=True, **kwargs):
        
        self.n_components = n_components
        
        self.normalization = normalization
        if self.normalization:
            self.normalization_standard = X.copy()
            X = normalize(X)
        
        self.mean_centric = mean_centric
        if self.mean_centric:
            self.x_mean = self.get_mean_initial(X)
            X = X - self.x_mean
        
        if standardization:
            self.scaler = StandardScaler().set_output(transform="pandas")
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = None
        
        super().__init__(X, **kwargs)
        
        return
    
    
    def get_mean_initial(self, X):
        return np.mean(X, axis=0, keepdims=True)
        
        
    def compute_pca(self, X, **kwargs):
        
        self.pca = PCA(n_components=self.n_components, **kwargs)
        # self.pca = FastICA(n_components=self.n_components, max_iter=1000, **kwargs)
        self.pca.fit(X)
        
        # self.components = self.pca.components_
        # self.mean = self.pca.mean_
        variances = self.pca.explained_variance_ratio_
        
        return variances, self.pca.components_.T, self.pca.mean_
    
    
    def transform(self, activations, device='cuda', **kwargs):
        
        activations_ = activations if isinstance(activations, np.ndarray) else activations.detach().cpu().numpy()
        if self.normalization:
            activations_ = normalize(activations_, normalization_standard=self.normalization_standard)
        if self.mean_centric:
            activations_ = activations_ - self.x_mean
        if self.scaler is not None:
            activations_ = self.scaler.transform(activations_)
        
        act_pca = self.pca.transform(activations_)
        # if isinstance(activations, np.ndarray):
        # else:
        #     activations = self.scaler.transform(activations.detach().cpu().numpy())
        #     act_pca = torch.tensor(self.pca.transform(activations.detach().cpu().numpy())).to(device)
        #     activations.to(device)
        
        return act_pca if isinstance(activations, np.ndarray) else torch.tensor(act_pca).to(device)
    
    
    def inverse_transform(self, act_pca, device='cuda', **kwargs):
        
        inverse_transform = self.pca.inverse_transform(act_pca) if isinstance(act_pca, np.ndarray) else self.pca.inverse_transform(act_pca.detach().cpu().numpy())
        if self.scaler is not None:
            inverse_transform = self.scaler.inverse_transform(inverse_transform)
        if self.mean_centric:
            inverse_transform = inverse_transform + self.x_mean
        if self.normalization:
            inverse_transform = de_normalize(inverse_transform, normalization_standard=self.normalization_standard)
        
        return inverse_transform if isinstance(act_pca, np.ndarray) else torch.tensor(inverse_transform).to(device)
    
    
    
class PCA_SKLEARN_MEDIAN(PCA_of_SKLEARN):
    def __init__(self, X, n_components=None, normalization:bool=False, standardization = False, mean_centric = True, **kwargs):
        super().__init__(X, n_components=n_components, normalization=normalization, standardization=standardization, mean_centric=mean_centric, **kwargs)
        return
    def get_mean_initial(self, X):
        _p = PCA(n_components=self.n_components)
        a_X = _p.fit_transform(X)
        a_X = np.median(a_X, axis=0, keepdims=True)
        X_r = _p.inverse_transform(a_X)
        return X_r
    
    
    
class Sparse_PCA_of_SKLEARN(PCA_of_SKLEARN):
    
    def __init__(self, X, n_components=None, **kwargs):
        super().__init__(X, n_components)
        
        
    def compute_pca(self, X, **kwargs):
        
        self.pca = SparsePCA(n_components=self.n_components, **kwargs)
        self.pca.fit(X)
        
        # self.components = self.pca.components_
        # self.mean = self.pca.mean_
        # self.variances = self.pca.explained_variance_ratio_
        
        return None, self.pca.components_.T, self.pca.mean_
    
    