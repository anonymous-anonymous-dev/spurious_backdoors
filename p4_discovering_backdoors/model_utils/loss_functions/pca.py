import numpy as np
import torch


from sklearn.decomposition import PCA



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
    
    def __init__(self, X):
        
        self.variances, self.components, self.mean = self.compute_pca(X)
        
        return
    
    
    def not_implemented(self):
        raise NotImplementedError('This function has not been implemented. Please call the child class.')
    
    
    def compute_pca(self, *args, **kwargs): return self.not_implemented()
    def transform(self, *args, **kwargs): return self.not_implemented()
    def inverse_transform(self, *args, **kwargs): return self.not_implemented()
    
    def reconstruct(self, X, highest_index: int=None):
        transformed_X = self.transform(X)
        return self.inverse_transform(transformed_X)
    
    

class PCA_of_NPCA(General_PCA):
    
    def __init__(self, X):
        
        super().__init__(X)
        
        return
    
    
    def compute_pca(self, X):
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
        sort_idcs = np.argsort(eig_vals)[::-1]

        return eig_vals[sort_idcs], eig_vecs[:, sort_idcs], X_mean
    
    
    def transform_numpy(self, activations, eigvec_scale: bool=False):
        act = activations - self.mean
        act_pca = act@self.components
        if eigvec_scale:
            act_pca = act_pca * np.sum(self.components, axis=0)
        return act_pca
    
    
    def transform_torch(self, activations: torch.Tensor, eigvec_scale: bool=False):
        
        device = activations.device
        act = activations.to(device) - torch.tensor(self.mean).to(device)
        act_pca = act@torch.tensor(self.components, dtype=torch.float32).to(device)
        if eigvec_scale:
            act_pca = act_pca * torch.sum(torch.tensor(self.components, dtype=torch.float32).to(device), dim=0)
        
        return act_pca
    
    
    def inverse_transform(self, act_pca):
        return (np.matmul(act_pca, self.components.T) + self.mean).astype(np.float32)



class PCA_of_SKLEARN(General_PCA):
    
    def __init__(self, X, n_components=None):
        
        self.n_components = None
        super().__init__(X)
        
        return
        
        
    def compute_pca(self, X):
        
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        
        # self.components = self.pca.components_
        # self.mean = self.pca.mean_
        # self.variances = self.pca.explained_variance_ratio_
        
        return self.pca.explained_variance_ratio_, self.pca.components_, self.pca.mean_
    

    def transform(self, activations):
        return self.pca.transform(activations)
    
    
    def inverse_transform(self, act_pca):
        return self.pca.inverse_transform(act_pca)
    
    