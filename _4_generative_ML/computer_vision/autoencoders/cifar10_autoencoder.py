import torch
from torch import nn

import gc


from transformers import AutoModel

from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput



class CIFAR_Autoenocder(nn.Module):
    
    def __init__(
        self,
        configuration: dict={},
        verbose: bool=False
    ):
        
        super().__init__()
        
        self.configuration = {
            'type': 'sd-vae-ft-mse-original', 
            'noise_mag': 0.25, 
            'n_iter':5, 
            'batch_size': 8,
            'device': 'cpu',
            'verbose': False
        }
        for key in configuration.keys():
            self.configuration[key] = configuration[key]
        
        # self.batch_size = configuration['batch_size']
        
        # Load model directly
        self.autoencoder = AutoModel.from_pretrained("nateraw/autoencoder-cifar10")
        
        for parameter in self.autoencoder.parameters(): parameter.requires_grd = False
        self.autoencoder.eval()
        self.eval()
        
        self.verbose = verbose
        
        return
    
    
    def forward(self, X):
        return self.autoencoder(X)
    
    
    def encoder_foward(self, X):
        return self.autoencoder.encode(X).latent_dist.mode()
    
    
    def decoder_forward(self, X):
        return DecoderOutput(sample=self.autoencoder.decode(X).sample)
    
    
    def batch_forward(self, X):
        
        device = X.device
        X = X.to('cpu')
        
        n_batches = len(X)//self.configuration['batch_size']
        n_batches += 0 if n_batches*self.configuration['batch_size']>=len(X) else 1
        
        outputs = []
        for i in range(n_batches):
            outputs.append(self.forward(X[i*self.configuration['batch_size']:(i+1)*self.configuration['batch_size']].to(device)).cpu())
            torch.cuda.empty_cache()
            gc.collect()
            
        return torch.cat(outputs, 0).to(device)
    
    