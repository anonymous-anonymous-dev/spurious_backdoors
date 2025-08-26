import numpy as np
import torch, torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from ...backdoor_attacks.simple_backdoor import Simple_Backdoor

from .backdoor_defense import Backdoor_Detection_Defense
from .neural_cleanse_official.neural_cleanse import Neural_Cleanse as Neural_Cleanse_Official



class Neural_Cleanse(Backdoor_Detection_Defense):
    """
    Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks
    URL: https://ieeexplore.ieee.org/abstract/document/8835365/
    
    @inproceedings{wang2019neural,
        title={Neural cleanse: Identifying and mitigating backdoor attacks in neural networks},
        author={Wang, Bolun and Yao, Yuanshun and Shan, Shawn and Li, Huiying and Viswanath, Bimal and Zheng, Haitao and Zhao, Ben Y},
        booktitle={2019 IEEE symposium on security and privacy (SP)},
        pages={707--723},
        year={2019},
        organization={IEEE}
    }
    """
    
    def __init__(self, torch_model: Torch_Model=None, defense_configuration: dict = {}, **kwargs):
        
        super().__init__(torch_model=torch_model, defense_configuration=defense_configuration)
        
        return
    
    
    def configure_defense(self, defense_configuration = ...):
        
        super().configure_defense(defense_configuration)
        
        self.official_neural_cleanse = Neural_Cleanse_Official(
            self.torch_model,
            configuration=self.configuration, 
            data_means=self.torch_model.data.data_means, 
            data_stds=self.torch_model.data.data_stds
        )
        
        return
    
    
    def defend(self, data_in: Simple_Backdoor, **kwargs):
        
        flag_label_list = self.official_neural_cleanse.analyze(data_in.test, target_labels=[0])
        
        return
    
    
    def prurify_model(self, model, *args, **kwargs):
        
        # TODO: Write purification code.
        
        return model
    
    