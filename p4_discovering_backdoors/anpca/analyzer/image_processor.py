import numpy as np
import torch
import math

from utils_.general_utils import normalize



class Image_PreProcessor:
    
    def __init__(self, thresh: float=0.1, device: str='cuda', batch_size: int=16):
        self.thresh = thresh
        self.device = device
        self.batch_size = batch_size
        return
    
    
    def torch_convolution_cuda_all_samples_as_one_batch(self, image: torch.tensor, kernel: torch.tensor, num_channels: int=3):
        if num_channels==3:
            return torch.cat([torch.nn.functional.conv2d(image[:, i:i+1].to('cuda'), kernel.to('cuda'), stride=1, padding='same').detach().cpu() for i in range(num_channels)], dim=1)
        return torch.nn.functional.conv2d(image.to('cuda'), kernel.to('cuda'), stride=1, padding='same').detach().cpu()
    
    
    def torch_convolution_cuda_batch_wise(self, image: torch.tensor, kernel: torch.tensor, num_channels: int=3):
        # Process the input in smaller batches along the sample (batch) dimension
        device = self.device #'cuda'
        bs = int(self.batch_size) if hasattr(self, 'batch_size') else 16
        n = image.shape[0]
        outputs = []
        k_cuda = kernel.to(device)
        for start in range(0, n, bs):
            end = min(n, start + bs)
            img_batch = image[start:end].to(device)
            if num_channels == 3:
                out_batch = torch.cat([
                    torch.nn.functional.conv2d(img_batch[:, i:i+1], k_cuda, stride=1, padding='same')
                    for i in range(num_channels)
                ], dim=1).detach().cpu()
            else:
                out_batch = torch.nn.functional.conv2d(img_batch, k_cuda, stride=1, padding='same').detach().cpu()
            outputs.append(out_batch)

        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)
    
    
    def torch_convolution_cuda(self, image: torch.tensor, kernel: torch.tensor, num_channels: int=3):
        return self.torch_convolution_cuda_all_samples_as_one_batch(image, kernel, num_channels=num_channels) if self.batch_size is None else self.torch_convolution_cuda_batch_wise(image, kernel, num_channels=num_channels)
    
    
    def get_edges(self, image: np.ndarray, thresh_in: float=-1, one_sided: bool=False):
        thresh_in = self.thresh if thresh_in<0 else thresh_in
        image = normalize(image)
        filter = 0.125 * np.array(
            [np.array([1,1,1]),
            np.array([1,-8,1]),
            np.array([1,1,1])]
        )
        filter = np.expand_dims(filter, axis=0)
        filter = np.expand_dims(filter, axis=0).astype(np.float32)
        output = self.torch_convolution_cuda(torch.tensor(np.mean(image, axis=1, keepdims=True)), torch.tensor(filter), num_channels=1).numpy()
        if not one_sided:
            output = np.abs(output)
            output = np.clip(output, 0, 1)
        if thresh_in is not None:
            output[output < thresh_in] = 0.
            output[output >= thresh_in] = 1.
        return output
    
    
    def smooth(self, image: np.ndarray, num_channels: int=3):
        filter = (1/9) * np.array(
            [
                np.array([1,1,1]),
                np.array([1,1,1]),
                np.array([1,1,1]),
                
            ]
        )
        filter = filter.reshape(1,1,3,3).astype(np.float32)
        output = image.copy()
        for i in range(1):
            output = self.torch_convolution_cuda(torch.tensor(output), torch.tensor(filter), num_channels=image.shape[1]).numpy()
        return output
    
    
    def remove_bad_edges(self, image_edges: np.ndarray):
        image_edges_smoothed = self.smooth(image_edges, num_channels=1)
        image_edges[image_edges_smoothed<=0.5] = 0
        image_edges[image_edges_smoothed<=0.1] = 1
        return image_edges
    
    
    def process_horizontal(self, image: np.ndarray, thresh_in: float=0.1, smoothing_iterations: int=3, recreate: bool=False, get_non_recreated: bool=False):
        output_smooth = image.copy()
        output_edges = self.get_edges(output_smooth, thresh_in=thresh_in, one_sided=False)
        # output_edges = self.remove_bad_edges(output_edges)
        output_edges = 1-output_edges
        # set random pixels in output edges to 1 in order to maintain color marks
        placeholders = np.zeros_like(output_edges[:, :1])
        placeholders[np.random.uniform(0, 1, size=placeholders.shape) < 0.8] = 1
        if image.shape[1]==3:
            output_edges = np.concatenate([output_edges, output_edges, output_edges], axis=1)
            placeholders = np.concatenate([placeholders, placeholders, placeholders], axis=1)
        # placeholders[output_edges==0] = 0    
        
        output_smooth_not_recreated = np.zeros_like(output_smooth)
        output_smooth_not_recreated[output_edges==1] = image[output_edges==1]
        
        if recreate:
            for i in range(smoothing_iterations):
                output_smooth = self.smooth(output_smooth); backup = output_smooth.copy()
                output_smooth[output_edges==1] = image[output_edges==1]
                output_smooth[placeholders==1] = image[placeholders==1]
            output_smooth_further = self.smooth(output_smooth.copy())
            output_smooth[placeholders==1] = output_smooth_further[placeholders==1]
        else:
            output_smooth = output_smooth_not_recreated.copy()
            
        if get_non_recreated:
            return np.clip(output_smooth, np.min(image), np.max(image)), np.clip(output_smooth_not_recreated, np.min(image), np.max(image))
        
        return np.clip(output_smooth, np.min(image), np.max(image))
    
    
    def process_wanet(self, image: np.ndarray, thresh_in: float=0.1, recreate: bool=False):
        output_smooth = image.copy()
        # output_edges = get_edges(output_smooth, thresh_in=0.05, one_sided=True)
        output_edges = self.get_edges(output_smooth, thresh_in=thresh_in, one_sided=False)
        # output_edges = self.remove_bad_edges(output_edges)
        output_edges = 1-output_edges
        # set random pixels in output edges to 1 in order to maintain color marks
        placeholders = np.zeros_like(output_edges[:, :1])
        placeholders[np.random.uniform(0, 1, size=placeholders.shape) < 0.8] = 1
        if image.shape[1]==3:
            output_edges = np.concatenate([output_edges, output_edges, output_edges], axis=1)
            placeholders = np.concatenate([placeholders, placeholders, placeholders], axis=1)
        # placeholders[output_edges==0] = 0
        if recreate:
            for i in range(5):
                output_smooth = self.smooth(output_smooth); backup = output_smooth.copy()
                output_smooth[output_edges==1] = image[output_edges==1]
                output_smooth[placeholders==1] = image[placeholders==1]
            output_smooth_further = self.smooth(output_smooth.copy())
            output_smooth[placeholders==1] = output_smooth_further[placeholders==1]
        else:
            output_smooth = np.zeros_like(output_smooth)
            output_smooth[output_edges==1] = image[output_edges==1]
        return np.clip(output_smooth, np.min(image), np.max(image))
    
    
    def local_random_skew(self, images, strength=0.02):
        images = torch.tensor(images).to('cpu')
        B, C, H, W = images.shape
        device = images.device
        # Base identity grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        # Random smooth displacement
        noise = torch.randn_like(grid) * strength
        grid = grid + noise
        return torch.nn.functional.grid_sample(images, grid, align_corners=False).detach().cpu().numpy()
    
    
    def process_invisible_perturbations(self, image: np.ndarray, thresh_in: float=0.1, recreate: bool=False):
        output_smooth = image.copy()
        output_edges = self.get_edges(output_smooth, thresh_in=thresh_in)
        # set random pixels in output edges to 1 in order to maintain color marks
        placeholders = np.zeros_like(output_edges[:, :1])
        placeholders[np.random.uniform(0, 1, size=placeholders.shape) < 0.7] = 1
        if image.shape[1]==3:
            output_edges = np.concatenate([output_edges, output_edges, output_edges], axis=1)
            placeholders = np.concatenate([placeholders, placeholders, placeholders], axis=1)
        # placeholders[output_edges==1] = 0
        for i in range(3):
            output_smooth = self.smooth(output_smooth); backup = output_smooth.copy()
            output_smooth[output_edges==1] = image[output_edges==1]
            output_smooth[placeholders==1] = image[placeholders==1]
        output_smooth_further = self.smooth(output_smooth.copy())
        output_smooth[placeholders==1] = output_smooth_further[placeholders==1]
        return np.clip(output_smooth, np.min(image), np.max(image))
    
    
    def process(self, image: np.ndarray):
        iwh = self.process_horizontal(image, recreate=False)
        iwi = self.process_invisible_perturbations(image)
        return iwh
    
    
    def rotate(self, images, degrees=5):
        images = torch.tensor(images)
        angles = torch.full((images.shape[0],), degrees, device=images.device)
        angles = angles * math.pi / 180.0

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        theta = torch.zeros(images.shape[0], 2, 3, device=images.device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        grid = torch.nn.functional.affine_grid(theta, images.size(), align_corners=False)
        return torch.nn.functional.grid_sample(images, grid, align_corners=False).detach().cpu().numpy()
    
    
    # def compute_loss_of_each_sample_in_dataloader(self, model: Torch_Model_Save_Best, dataloader: torch.utils.data.DataLoader, loss_function):
        
    #     losses = []
    #     model.model.eval()
    #     with torch.no_grad():
    #         for batch_idx, (inputs, targets) in enumerate(dataloader):
    #             inputs, targets = inputs.to(model.device), targets.to(model.device)
    #             outputs = model.model(inputs)
    #             batch_losses = loss_function(outputs, targets, reduction='none')
    #             losses.extend(batch_losses.cpu().numpy())
                
    #     return np.array(losses)
    
    