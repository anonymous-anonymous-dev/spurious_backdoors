import torch, torchvision
import numpy as np
from copy import deepcopy

from PIL import Image

from diffusers import StableDiffusionUpscalePipeline


from _0_general_ML.model_utils.torch_model import Torch_Model

from ...backdoor_attacks.simple_backdoor import Simple_Backdoor
from .backdoor_defense import Backdoor_Detection_Defense

from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy



class Zero_Shot_Image_Purification(Backdoor_Detection_Defense):
    
    def __init__(self, torch_model: Torch_Model=None, defense_configuration: dict={}, **kwargs):
        
        super().__init__(torch_model, defense_configuration, **kwargs)
        
        return
    
    
    def configure_defense(self, *args, defense_configuration: dict={}, **kwargs):
        
        default_configuration = {
            'diffusion_batch_size': 32,
            'diffusion_iterations': 5
        }
        for key in default_configuration.keys():
            if key not in defense_configuration.keys():
                defense_configuration[key] = default_configuration[key]
        
        super().configure_defense(*args, defense_configuration=defense_configuration, **kwargs)
        
        self.diffusion_batch_size = self.configuration['diffusion_batch_size']
        self.means = self.torch_model.data.data_means
        self.stds = self.torch_model.data.data_stds
        
        return
    
    
    def defend(self, *args, **kwargs):
        
        # --- Load SR3-style model from Hugging Face ---
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            revision="fp16",
            torch_dtype=torch.float16
        ).to("cuda")
        
        self.to_pil = torchvision.transforms.ToPILImage()
        self.original_size = self.torch_model.data.preferred_size
        if isinstance(self.original_size, int):
            self.original_size = (self.original_size, self.original_size)
        self.smaller_size = (int(0.79*self.original_size[0]), int(0.79*self.original_size[1]))
        
        # --- SR3 expects 64x64 low-res input (scaled image) ---
        self.desize_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.smaller_size, interpolation=Image.BICUBIC),
            torchvision.transforms.ToPILImage()
        ])
        self.recons_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.original_size, interpolation=Image.BICUBIC)
        ])
        
        
        self.final_model = self.torch_model
        
        return
    
    
    def denormalize(self, inputs):
        return (inputs * torch.tensor(self.stds).reshape(1, -1, 1, 1)) + torch.tensor(self.means).reshape(1, -1, 1, 1)
    def renormalize(self, inputs):
        return (inputs - torch.tensor(self.means).reshape(1, -1, 1, 1)) / torch.tensor(self.stds).reshape(1, -1, 1, 1)
    
    
    def purify_input(self, x):
        low_res_pil = [self.desize_transform(img) for img in x]
        prompt = "a clean and sharp photo"
        purified_imgs = []
        for i in range(self.configuration['diffusion_iterations']):
            high_res_pil = self.pipe(prompt=[prompt]*len(low_res_pil), image=low_res_pil)["images"]
            _purified_imgs = torch.stack([self.recons_transform(img) for img in high_res_pil], dim=0)
            purified_imgs.append(_purified_imgs)
        purified_imgs = torch.mean(torch.stack(purified_imgs, dim=0), dim=0)
        
        return purified_imgs.detach().cpu().numpy()
    
    
    def batch_wise_purify(self, x: np.ndarray):
        # check channels
        channels = 3
        if x.shape[1]==1:
            channels = 1
            x = np.concatenate([x, x, x], axis=1)
        num_batches = len(x) // self.diffusion_batch_size
        num_batches += 1 if self.diffusion_batch_size*num_batches<len(x) else 0
        purified_xs = []
        for i in range(num_batches):
            print(f'\rPurifying: {i+1}/{num_batches}.', end='')
            purified_batch = self.purify_input(torch.tensor(x.astype(np.float32)[i*self.diffusion_batch_size:(i+1)*self.diffusion_batch_size]))
            purified_xs.append(purified_batch)
        purified_xs = np.concatenate(purified_xs, axis=0)
        if channels==1:
            return np.mean(purified_xs, axis=1, keepdims=True)
        return purified_xs
    
    
    def evaluate(self, data_in: Simple_Backdoor, *args, number_of_items = 100, **kwargs):
        
        clean_dataloader = torch.utils.data.DataLoader(data_in.test, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False)
        poisoned_dataloader = torch.utils.data.DataLoader(data_in.poisoned_test, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False)
        
        cx, cy = get_data_samples_from_loader(clean_dataloader, size=number_of_items)
        cx = self.denormalize(cx).numpy()
        cx = self.batch_wise_purify(cx[:number_of_items])
        cx = self.renormalize(torch.tensor(cx)).numpy()
        cdl = prepare_dataloader_from_numpy(cx, cy[:number_of_items], batch_size=32)
        
        px, py = get_data_samples_from_loader(poisoned_dataloader, size=number_of_items)
        px = self.denormalize(px).numpy()
        px = self.batch_wise_purify(px[:number_of_items])
        px = self.renormalize(torch.tensor(px)).numpy()
        pdl = prepare_dataloader_from_numpy(px, py[:number_of_items], batch_size=32)
        
        loss_clean, acc_clean, _ = self.torch_model.test_shot(cdl); print()
        loss_poisoned, acc_poisoned, _ = self.torch_model.test_shot(pdl)
        
        return (loss_clean, acc_clean), (loss_poisoned, acc_poisoned)
    
    
    def __purify_model(self, model: torch.nn.Module):
        
        class Purified_Net(torch.nn.Module):
            
            def __init__(local_self, net: torch.nn.Module):
                super().__init__()
                local_self.net = net
                return
            
            def purify_input(local_self, x):
                lowres_resized = torch.nn.functional.interpolate(x, size=self.smaller_size, mode='bilinear', align_corners=False)
                # ---- SR3 prompt (can be arbitrary) ----
                prompt = ["a sharp photo"] * len(x)
                # ---- Super-resolution with SR3 (result: 256x256) ----
                # Pipeline returns list of PIL images, but it internally handles tensors
                with torch.autocast("cuda"):
                    outputs = self.pipe(prompt=prompt, image=lowres_resized)["images"]
                # ---- Convert to torch tensors (normalized to [0, 1]) ----
                # This is necessary since outputs are PIL images
                to_tensor = torchvision.transforms.ToTensor()
                highres_tensor = torch.stack([to_tensor(img) for img in outputs]).to(self.torch_model.device)
                # ---- Resize to (124x124) if desired ----
                highres_124 = torch.nn.functional.interpolate(highres_tensor, size=self.original_size, mode='bilinear', align_corners=False)
                return highres_124
            
            def forward(local_self, x):
                x = local_self.purify_input(x)
                return local_self.net(x)
        
        model = Purified_Net(model)
        
        return model
    
    
    def __purify_image(self, x: torch.Tensor):
        
        low_res_pil = [self.to_pil(img) for img in x]
        
        # Resize to 64x64 (input to SR3)
        low_res_pil = [img.resize(self.smaller_size, resample=Image.BICUBIC) for img in low_res_pil]
        
        # --- Prompt required (even if empty) ---
        prompt = "a clean and sharp photo"
        
        # --- Super-resolve images ---
        high_res_pil = []
        for img in low_res_pil:
            output = self.pipe(prompt=prompt, image=img).images[0]
            high_res_pil.append(output)
        
        purified_imgs = np.array([np.array(img.resize(self.original_size)).transpose(2,0,1).astype(np.float32)/255 for img in high_res_pil])
        purified_imgs = torch.tensor(purified_imgs)
        
        return purified_imgs
    
    
    def __purify_image(self, input_tensor: torch.Tensor):
        
        downscaled_tensor_by_size = torch.nn.functional.interpolate(input_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        
        # model_id = "stabilityai/stable-diffusion-x4-upscaler"
        # pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        # pipeline = pipeline.to("cuda") # or "cpu" if no GPU
        
        # 3. Upscale
        prompt = "a clean and sharp photo" # Optional: add a descriptive prompt
        upscaled_image = self.pipe(prompt=prompt, image=downscaled_tensor_by_size).images[0]
        
        return upscaled_image
    
    
    