import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model


class Parameters:
    
    def __init__(
        self, data: Torch_Dataset,
        attack_mode: str='all2one', target_class: int=0,
        pc: float=0.1, cross_ratio: float=0.,
        s: float=0.5, k: int=4, grid_rescale: float=1,
        device: str='cuda'
    ):
        
        self.data = data
        self.get_data_stats()
        
        self.pc = pc
        self.cross_ratio = cross_ratio
        self.s = s
        self.k = k
        self.attack_mode = attack_mode
        self.target_label = target_class
        self.grid_rescale = grid_rescale
        self.device = device
        
        self.schedulerC_lambda = 0.1
        self.schedulerC_milestones = [100, 200, 300, 400]
        self.n_iters= 1000
        
        return
    
    
    def get_data_stats(
        self
    ):
        
        (self.input_height, self.input_width) = self.data.preferred_size
        self.input_channel = self.data.train.__getitem__(0)[0].shape[0]
        self.num_classes = self.data.num_classes
        
        self.data_means = self.data.data_means
        self.data_stds = self.data.data_stds
        
        self.dataset = self.data.data_name
        
        return