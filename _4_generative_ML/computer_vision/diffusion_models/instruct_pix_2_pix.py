import torch

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


from .diffuser_model import Diffuser_Model



my_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Instruct_Pix_2_Pix(Diffuser_Model):
    
    def __init__(
        self, 
        model_configuration: dict = {}, 
        results_path: str = '', 
        model_name: str = '', 
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
        
        self.model_name = 'timbrooks/instruct-pix2pix' if self.model_name=='' else self.model_name
        self.device = my_device
        
        return
    
    
    def prepare_model(self, device=None):
        
        self.device = my_device if device is not None else self.device
        
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(self.model_name, torch_dtype=torch.float16, safety_checker=None).to(self.device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # self.prompt = "turn this into a hummingbird"
        
        return
    
    