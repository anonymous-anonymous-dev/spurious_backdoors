import torch


from _0_general_ML.model_utils.torch_model import Torch_Model


class Parameters:
    
    def __init__(
        self,
        torch_model: Torch_Model
    ):
        
        self.get_data_stats(torch_model)
        
        self.pc = 0.1
        self.device = 'cuda'
        self.cross_ratio = 0. #2    # rho_a = pc, rho_n = pc * cross_ratio
        self.attack_mode = 'all2one'
        self.target_label = 0
        self.schedulerC_lambda = 0.1
        self.schedulerC_milestones = [100, 200, 300, 400]
        self.k = 4
        self.s = 0.5
        self.n_iters= 1000
        self.grid_rescale = 1
        
        return
    
    
    def get_data_stats(
        self, torch_model: Torch_Model
    ):
        
        (self.input_height, self.input_width) = torch_model.data.preferred_size
        self.input_channel = torch_model.data.train.__getitem__(0)[0].shape[0]
        self.num_classes = torch_model.data.num_classes
        
        self.data_means = torch_model.data.data_means
        self.data_stds = torch_model.data.data_stds
        
        self.dataset = torch_model.data.data_name
        
        return