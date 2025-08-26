import torch
import os


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _4_generative_ML.computer_vision.autoencoders.denoised_autoencoder import Denoising_Autoencoder



def get_autoencoder(autoencoder_configuration: dict, verbose: bool=False) -> Denoising_Autoencoder:
    
    autoencoder_class_dict = {
        'sd-vae-ft-mse-original': Denoising_Autoencoder
    }
    
    return autoencoder_class_dict[autoencoder_configuration['type']](autoencoder_configuration, verbose=verbose)


class Robustify_Model(torch.nn.Module):
    
    def __init__(
        self, 
        classifier: torch.nn.Module, 
        autoencoder_configuration: dict={},
        verbose: bool=True
    ):
        
        super().__init__()
        
        self.classifier = classifier
        
        self.autoencoder_configuration = {
            'type': 'sd-vae-ft-mse-original', 
            'noise_mag': 0.25, 
            'n_iter':5, 
            'batch_size': 8,
            'device': 'cuda'
        }
        for key in autoencoder_configuration.keys():
            self.autoencoder_configuration[key] = autoencoder_configuration[key]
        self.autoencoder = get_autoencoder(self.autoencoder_configuration, verbose=verbose)
        
        self.pca_transormation = False
        self.verbose = verbose
        
        return
    
    
    def forward(self, X):
        return self.classifier(self.autoencoder(X))
    
    