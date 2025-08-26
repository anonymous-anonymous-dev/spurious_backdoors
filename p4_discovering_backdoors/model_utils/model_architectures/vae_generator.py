import numpy as np
import torch, torchvision
from PIL import Image
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VAE_Gen(torch.nn.Module):
    
    def __init__(
        self, 
        configuration: dict={},
        verbose: bool=False,
        **kwargs
    ):
        
        super().__init__()
        
        self.configuration = {
            'type': 'CompVis/stable-diffusion-v1-4',
            'height': 512,
            'width': 512,
            'channels': 4,
            'num_inference_steps': 100,
            'scaling_factor': 1 / 0.18215,
            'batch_size': 4,
            'outsize': 512
        }
        for key in configuration.keys():
            self.configuration[key] = configuration[key]
            
        # self.guidance_scale = 7.5                # Scale for classifier-free guidance
        self.generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise
        self.prepare_pipeline()
        
        self.verbose = verbose
        
        return
    
    
    def prepare_pipeline(self, device=torch_device):
        
        self.device = device
        self.model = AutoencoderKL.from_pretrained(self.configuration['type'], subfolder="vae").to(self.device)
        
        return
    
    
    def generate_random_latents(self):
        
        latents = torch.randn(
            (
                self.configuration['batch_size'], 
                self.configuration['channels'], self.configuration['height'] // 8, self.configuration['width'] // 8
            ),
            generator=self.generator,
        )
        latents = latents.to(torch_device)
        
        return latents #* self.scheduler.init_noise_sigma
    
    
    def forward(self, X): 
        return torchvision.transforms.functional.resize(
            self.model.decode(X*self.configuration['scaling_factor']).sample, 
            (self.configuration['outsize'],)
        )
    
    
    def generate_image_from_vae(self, latents):
        
        with torch.no_grad():
            image = self(latents)
            
        return image
    
    
    def get_pil_image_from_torch_images(self, image):
        
        # the image is finally converted to pil image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        return pil_images
    
    
    def adv_loss_outputs(self, y_1, y_2, **kwargs): return torch.mean(torch.square(y_1-y_2))
    
    
    def deprecated_update_latent(self, latent, y_input, targeted: bool=True):
        
        y_in_s = torch.tensor(y_input).to(self.device)
        latent_in = torch.tensor(latent.astype(np.float32)).to(self.device)
        latent_delta = torch.autograd.Variable(torch.tensor(latent)).to(self.device)
        latent_delta.requires_grad=True
        
        loss = 0
        for m, y_in in enumerate(y_in_s):
            prediction = self.model.decode(latent_in+latent_delta)
            if targeted:
                classification_loss = self.adv_loss_outputs(prediction, y_in)
            else:
                classification_loss = -1 * self.adv_loss_outputs(prediction, y_in)
            loss += classification_loss # + (1-self.alpha)*(torch.mean(torch.square(x_delta)))
        torch.mean(loss).backward()
        
        self.classification_loss_str = f'c_loss: {torch.mean(classification_loss)}'
        self.adversarial_loss_str = f''
        
        return latent_delta.grad.data.detach().cpu(), torch.mean(loss)
    
    
    def deprecated_step(self, latent_in, feature, latent_delta, epsilon=0.05, targeted: bool=False, **kwargs):
        
        no_of_batches = int(len(latent_in) / self.batch_size) + 1
        
        latent_delta_s, loss_s = [], []
        for batch_number in range(no_of_batches):
            start_index = batch_number * self.batch_size
            end_index = min( (batch_number+1)*self.batch_size, len(latent_in) )
            
            latent_delta_grad, loss_ = self.compute_loss(
                latent_in[start_index:end_index], feature,
                latent_delta, targeted=targeted
            )
            
            x_delta_s.append(x_delta_grad); loss_s.append(loss_)
        
        x_perturbation -= epsilon * torch.mean(torch.stack(x_delta_s, 0), 0).sign().numpy()
        
        self.last_run_loss_values.append(torch.mean(torch.stack(loss_s, 0)).item())
        
        return x_perturbation
    
    
    