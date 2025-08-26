import torch
import os


from .model_architectures.vae_generator import VAE_Gen



def get_latent_model(latent_model_configuration: dict, verbose: bool=False) -> VAE_Gen:
    
    latent_model_class_dict = {
        'CompVis/stable-diffusion-v1-4': VAE_Gen
    }
    
    return latent_model_class_dict[latent_model_configuration['type']](latent_model_configuration, verbose=verbose)


class Latentify_Model(torch.nn.Module):
    
    def __init__(
        self, 
        classifier: torch.nn.Module, 
        latent_model_configuration: dict={},
        verbose: bool=True
    ):
        
        super().__init__()
        
        self.classifier = classifier
        
        self.latent_model_configuration = {
            'type': 'CompVis/stable-diffusion-v1-4', 
            'batch_size': 8,
            'device': 'cuda'
        }
        for key in latent_model_configuration.keys():
            self.latent_model_configuration[key] = latent_model_configuration[key]
        self.latent_model = get_latent_model(self.latent_model_configuration, verbose=verbose)
        
        self.pca_transormation = False
        self.verbose = verbose
        
        return
    
    
    def forward(self, X):
        return self.classifier(self.latent_model(X))
    
    