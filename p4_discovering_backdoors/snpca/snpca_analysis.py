import torch
import numpy as np
from copy import deepcopy

from sklearn.cluster import KMeans


from _4_generative_ML.computer_vision.autoencoders.denoised_autoencoder import Denoising_Autoencoder

from ..model_utils.quantizer import Quantization

from utils_.pca import PCA_of_SKLEARN, PCA_of_NPCA, PCA_SKLEARN_MEDIAN
from utils_.general_utils import normalize
from utils_.visual_utils import show_image_grid
from utils_.torch_utils import get_data_samples_from_loader, get_outputs, prepare_dataloader_from_numpy




class Denoising_Autoencoder_Custom(Denoising_Autoencoder):
    
    def __init__(self, configuration = ..., verbose = False):
        super().__init__(configuration, verbose)
        self.device = self.configuration['device']
        return
    
    def forward(self, X, **kwargs):
        return self.encoder_foward(X.to('cpu'), **kwargs)



class SNPCA_Analysis:
    
    def __init__(self, quantization_levels: int=150, feature_model: torch.nn.Module=None):
        
        self.feature_model = feature_model
        if self.feature_model is None:
            self.feature_model = Denoising_Autoencoder_Custom(
                configuration={
                    'noise_mag': 0, 
                    'n_iter': 0,
                    'device': 'cuda',
                    'interpolation_mag': 0.4
                }
            )
        self.feature_model.eval()
            
        self.quantizer = Quantization(quantization_levels=quantization_levels, quantization_hardness=50)
            
        return
    
    
    def sample_scores(self, dataloader: torch.utils.data.DataLoader):
        scores_1 = self.analyze_large(dataloader)
        scores_2 = self.analyze_small(dataloader)
        return scores_1 + scores_2
    
    
    def input_scores_small(self, pca_: PCA_of_NPCA, inputs: np.ndarray, reconstruct_with_normalization: bool=True, return_numpy=False, **kwargs):
        
        components = pca_.transform(inputs)
        features = np.clip(components, 0, np.max(components))
        # features = np.sign(features)
        # features = np.abs(components)
        features = np.mean(features, axis=1)
        
        # recon_in = pca_.reconstruct(inputs)
        # # recon_in *= 0.
        # # recon_in = pca_.inverse_transform(recon_in)
        
        # if reconstruct_with_normalization:
        #     recon_in = np.clip(recon_in, np.min(inputs), np.max(inputs))
        #     features = np.mean((normalize(recon_in, normalization_standard=inputs)-normalize(inputs))**2, axis=1)
        #     # features = np.mean((recon_in-inputs)**2, axis=1)
        # else:
        #     features = np.mean((recon_in-inputs)**2, axis=1)
        
        return features
    
    
    def input_scores_large(self, pca_: PCA_of_NPCA, inputs: np.ndarray, reconstruct_with_normalization: bool=True, return_numpy=False, **kwargs):
        
        # components = pca_.transform(inputs)
        # features = np.clip(components, 0, np.max(components))
        # # features = np.sign(features)
        # # features = np.abs(components)
        # features = np.mean(features, axis=1)
        
        recon_in = pca_.reconstruct(inputs)
        # recon_in *= 0.
        # recon_in = pca_.inverse_transform(recon_in)
        
        if reconstruct_with_normalization:
            recon_in = np.clip(recon_in, np.min(inputs), np.max(inputs))
            features = np.mean((normalize(recon_in, normalization_standard=inputs)-normalize(inputs))**2, axis=1)
            # features = np.mean((recon_in-inputs)**2, axis=1)
        else:
            features = np.mean((recon_in-inputs)**2, axis=1)
        
        return features
    
    
    def torch_normalize(self, x: torch.tensor):
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))
    
    
    def __analyze_small(self, dataloader: torch.utils.data.DataLoader):
        
        x, y = get_data_samples_from_loader(dataloader)
        
        x_small_values = x - self.quantizer(x)
        x_normalized_small = self.torch_normalize(x_small_values)
        
        self.feature_model.mode = 'only_activations'
        with torch.no_grad():
            x_features = self.feature_model(x_normalized_small.to(self.feature_model.device), verbose=True).detach()
        self.feature_model.mode = 'default'
        
        xr_ = x_features.cpu().numpy().reshape(len(x_features), -1)
        
        repititions = 1000
        subset_population = 50
        sub_pca_components = 2
        scores = []
        all_components = []
        all_reconstructed = []
        for i in range(repititions):
            print(f'\rIteration {i+1}/{repititions}', end='')
            # randomly sample sample_num samples from inputs
            indices = np.random.choice(range(xr_.shape[0]), subset_population, replace=False)
            
            # prepare new pca on activations
            new_pca = PCA_SKLEARN_MEDIAN(xr_[indices], n_components=sub_pca_components)
            
            all_components.append(new_pca.components)
            all_reconstructed.append(new_pca.reconstruct(xr_))
            
            # compute input scores of sampled inputs
            scores.append(self.input_scores_small(new_pca, xr_, reconstruct_with_normalization=True))
            
        mean_scores = np.mean(np.array(scores), axis=0)
        
        # difference from mean. This seems to work
        # xr_ = xr_ - np.median(xr_, axis=0, keepdims=True)
        # mean_scores = np.mean(xr_**2, axis=1)
        # plt.hist(diff_[mean_scores.clean_indices], alpha=0.5, bins=100, label='clean');
        # plt.hist(diff_[mean_scores.poison_indices], alpha=0.5, bins=100, label='poisoned');
        
        return mean_scores
    
    
    def analyze_small(self, dataloader: torch.utils.data.DataLoader):
        
        x, y = get_data_samples_from_loader(dataloader)
        
        x_small_values = x - self.quantizer(x)
        x_normalized_small = self.torch_normalize(x_small_values)
        
        loader = prepare_dataloader_from_numpy(x_normalized_small, y, batch_size=dataloader.batch_size, shuffle=False)
        
        return self.analyze_large(loader)
        
        
    def analyze_large(self, dataloader: torch.utils.data.DataLoader):
        
        x_normalized_small, y = get_data_samples_from_loader(dataloader)
        
        self.feature_model.mode = 'only_activations'
        with torch.no_grad():
            x_features = self.feature_model(x_normalized_small.to(self.feature_model.device), verbose=True).detach()
        self.feature_model.mode = 'default'
        
        xr_ = x_features.cpu().numpy().reshape(len(x_features), -1)
        
        repititions = 1000
        subset_population = 50
        sub_pca_components = 2
        scores = []
        all_components = []
        all_reconstructed = []
        for i in range(repititions):
            print(f'\rIteration {i+1}/{repititions}', end='')
            # randomly sample sample_num samples from inputs
            indices = np.random.choice(range(xr_.shape[0]), subset_population, replace=False)
            
            # prepare new pca on activations
            new_pca = PCA_SKLEARN_MEDIAN(xr_[indices], n_components=sub_pca_components)
            
            all_components.append(new_pca.components)
            all_reconstructed.append(new_pca.reconstruct(xr_))
            
            # compute input scores of sampled inputs
            scores.append(self.input_scores_small(new_pca, xr_, reconstruct_with_normalization=True))
            
        mean_scores = np.mean(np.array(scores), axis=0)
        
        # difference from mean. This seems to work
        # xr_ = xr_ - np.median(xr_, axis=0, keepdims=True)
        # mean_scores = np.mean(xr_**2, axis=1)
        # plt.hist(diff_[mean_scores.clean_indices], alpha=0.5, bins=100, label='clean');
        # plt.hist(diff_[mean_scores.poison_indices], alpha=0.5, bins=100, label='poisoned');
        
        return mean_scores
    
    
    def __analyze_large(self, dataloader: torch.utils.data.DataLoader):
        
        x, y = get_data_samples_from_loader(dataloader)
        
        x_normalized = x
        # x_normalized = self.torch_normalize(x)
        # x_normalized = self.quantizer(x_normalized)
        
        self.feature_model.mode = 'only_activations'
        with torch.no_grad():
            x_features = self.feature_model(x_normalized.to(self.feature_model.device), verbose=True).detach()
        self.feature_model.mode = 'default'
        
        xr_ = x_features.cpu().numpy().reshape(len(x_features), -1)
        
        repititions = 1000
        subset_population = 50
        sub_pca_components = 2
        scores = []
        all_components = []
        all_reconstructed = []
        for i in range(repititions):
            print(f'\rIteration {i+1}/{repititions}', end='')
            # randomly sample sample_num samples from inputs
            indices = np.random.choice(range(xr_.shape[0]), subset_population, replace=False)
            
            # prepare new pca on activations
            new_pca = PCA_of_SKLEARN(xr_[indices], n_components=sub_pca_components)
            
            all_components.append(new_pca.components)
            all_reconstructed.append(new_pca.reconstruct(xr_))
            
            # compute input scores of sampled inputs
            scores.append(self.input_scores_small(new_pca, xr_, reconstruct_with_normalization=True))
            
        mean_scores = np.mean(np.array(scores), axis=0)
        
        # difference from mean. This seems to work
        # xr_ = xr_ - np.median(xr_, axis=0, keepdims=True)
        # mean_scores = np.mean(xr_**2, axis=1)
        # plt.hist(diff_[mean_scores.clean_indices], alpha=0.5, bins=100, label='clean');
        # plt.hist(diff_[mean_scores.poison_indices], alpha=0.5, bins=100, label='poisoned');
        
        return mean_scores
    
    