import numpy as np
import torch


from _0_general_ML.model_utils.torch_model import Torch_Model



class RegressionModel(torch.nn.Module):
    
    def __init__(
        self, 
        torch_model: Torch_Model, 
        init_mask, init_pattern, 
        data_means: list[int]=[0], data_stds: list[int]=[1], 
        epsilon: float=1e-7, normalization: bool=True
    ):
        
        self._EPSILON = epsilon
        super(RegressionModel, self).__init__()
        self.mask_tanh = torch.nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = torch.nn.Parameter(torch.tensor(init_pattern))
        
        self.torch_model = torch_model
        self.classifier = self.torch_model.model
        # self.normalizer = self._get_normalize(opt)
        # self.denormalizer = self._get_denormalize(opt)
        
        self.data_means = np.array(data_means).astype(np.float32).reshape(-1, 1, 1)
        self.data_stds = np.array(data_stds).astype(np.float32).reshape(-1, 1, 1)
        
        self.normalization = normalization
        
        return

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalization:
            pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = torch.nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5
    
    def get_raw_pattern(self):
        pattern = torch.nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5
    
    def normalizer(self, x):
        # print(x.shape, self.data_means.shape, self.data_stds.shape)
        if isinstance(x, np.ndarray):
            return (x - self.data_means) / self.data_stds
        return (x - torch.tensor(self.data_means).to(x.device)) / torch.tensor(self.data_stds).to(x.device)
    
    
    
class Recorder:
    def __init__(self, init_cost: float, cost_multiplier: float, **kwargs):
        
        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = init_cost
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        
        return

    def reset_state(self, init_cost: float=1e-3):
        self.cost = init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))
        return
    
    