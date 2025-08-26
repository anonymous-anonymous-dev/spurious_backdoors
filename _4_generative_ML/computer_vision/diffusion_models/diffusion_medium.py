import torch

from diffusers import StableDiffusion3Pipeline, DiffusionPipeline


from .diffuser_model import Diffuser_Model



my_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Stable_Diffusion(Diffuser_Model):
    
    def __init__(
        self, 
        model_configuration: dict = {}, 
        results_path: str = '', 
        model_name: str = '', 
        tokenizer_name: str = None, 
        verbose: bool = True, 
        **kwargs
    ):
        
        super().__init__(
            model_configuration=model_configuration,
            results_path=results_path, 
            model_name=model_name, 
            verbose=verbose, 
            **kwargs
        )
        
        self.model_name = 'stabilityai/stable-diffusion-3-medium-diffusers' if self.model_name=='' else self.model_name
        self.device = my_device
        
        return
    
    
    def configure_model(self):
        
        self.pipeline = DiffusionPipeline.from_pretrained("")
        
        return
    
    
    def prepare_model(self, device=None, callback_fn=None):
        
        self.device = my_device if device is not None else self.device
        
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.model_name, 
            callback_on_step_end=callback_fn,
            torch_dtype=torch.float16
        ).to(self.device)
        
        return
    
    
    def call(self, prompt='A cat holding a sign that says hello world'):

        image = self.pipeline(
            prompt, negative_prompt="", num_inference_steps=28, guidance_scale=7.0,
        ).images[0]
        
        return image
    
    
    