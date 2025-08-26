import torch

from diffusers import StableDiffusionXLImg2ImgPipeline


from .diffuser_model import Diffuser_Model



my_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Stable_Diffusion_Refiner(Diffuser_Model):
    
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
            tokenizer_name=tokenizer_name, 
            verbose=verbose, 
            **kwargs
        )
        
        self.model_name = 'stabilityai/stable-diffusion-xl-refiner-1.0' if self.model_name=='' else self.model_name
        self.device = my_device
        
        return
    
    
    def prepare_model(self, device=None):
        
        self.device = my_device if device is not None else self.device
        
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(self.model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(self.device)
        # self.prompt = "a hummingbird"
        
        return
    
    