import torch
from torch import nn

import gc

from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput



class Denoising_Autoencoder(nn.Module):
    
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
        
        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
        self.autoencoder = AutoencoderKL.from_single_file(url).to(self.configuration['device'])
        
        for parameter in self.autoencoder.parameters(): parameter.requires_grd = False
        self.autoencoder.eval()
        self.eval()
        
        self.verbose = verbose
        
        return
    
    
    def forward(self, X):
        
        _range = self.configuration['noise_mag'] * ( torch.max(X)-torch.min(X) )
        
        outputs = self.autoencoder(X)[0] 
        with torch.no_grad():
            for i in range(self.configuration['n_iter']):
                outputs += self.autoencoder( X+_range*torch.randn_like(X) )[0]
            
        return torch.div(outputs, self.configuration['n_iter']+1)
    
    
    def encoder_foward(self, X, verbose: bool=False, **kwargs):
        
        device = X.device
        my_device = self.configuration['device']
        X = X.to('cpu')
        
        n_batches = len(X)//self.configuration['batch_size']
        n_batches += 0 if n_batches*self.configuration['batch_size']>=len(X) else 1
        
        outputs = []
        for i in range(n_batches):
            if verbose: print(f'\rEncoding [{i+1}/{n_batches}].', end='')
            outputs.append(self.autoencoder.encode(X[i*self.configuration['batch_size']:(i+1)*self.configuration['batch_size']].to(my_device)).latent_dist.mode().cpu())
            torch.cuda.empty_cache()
            gc.collect()
        
        return torch.cat(outputs, 0).to(device)
    
    
    def decoder_forward(self, X,  verbose: bool=False, **kwargs):
        
        device = X.device
        my_device = self.configuration['device']
        X = X.to('cpu')
        
        n_batches = len(X)//self.configuration['batch_size']
        n_batches += 0 if n_batches*self.configuration['batch_size']>=len(X) else 1
        
        outputs = []
        for i in range(n_batches):
            if verbose: print(f'\rDecoding [{i+1}/{n_batches}].', end='')
            outputs.append(
                DecoderOutput(
                    sample = self.autoencoder.decode(
                        X[i*self.configuration['batch_size']:(i+1)*self.configuration['batch_size']].to(my_device)
                    ).sample
                )[0].cpu()
            )
            torch.cuda.empty_cache()
            gc.collect()
        
        return torch.cat(outputs, 0).to(device)
    
    
    def batch_forward(self, X, verbose: bool=False, **kwargs):
        
        device = X.device
        my_device = self.configuration['device']
        X = X.to('cpu')
        
        n_batches = len(X)//self.configuration['batch_size']
        n_batches += 0 if n_batches*self.configuration['batch_size']>=len(X) else 1
        
        outputs = []
        for i in range(n_batches):
            if verbose: print(f'\rEncoding [{i+1}/{n_batches}].', end='')
            outputs.append(self(X[i*self.configuration['batch_size']:(i+1)*self.configuration['batch_size']].to(my_device)).cpu())
            torch.cuda.empty_cache()
            gc.collect()
            
        return torch.cat(outputs, 0).to(device)
    
    