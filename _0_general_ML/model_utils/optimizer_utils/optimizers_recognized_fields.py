from copy import deepcopy


class Revised_CFGs:
    def __init__(self, optimizer_dict: dict):
        self.optimizer_dict = deepcopy(optimizer_dict)
        return
    
    
    def get_revised_dict(self, **cfg):
        
        fns = {
            'adamw': self.get_adam_cfg,
            'adam': self.get_adam_cfg,
            'sgd': self.get_sgd_cfg
        }
        
        return fns[self.optimizer_dict['name']](**cfg)
    
        
    def get_sgd_cfg(self, **cfg):
        recognized_fields = [
            'lr', 'weight_decay', 'momentum'
        ]
        return {k: v for k, v in cfg.items() if k in recognized_fields}


    def get_adam_cfg(self, **cfg):
        recognized_fields = [
            'lr', 'weight_decay'
        ]
        return {k: v for k, v in cfg.items() if k in recognized_fields}


    def get_adamw_cfg(self, **cfg):
        recognized_fields = [
            'lr', 'weight_decay'
        ]
        return {k: v for k, v in cfg.items() if k in recognized_fields}


    def __get_adamw_cfg(self, **cfg):
        recognized_fields = [
            'lr', 'weight_decay'
        ]
        return {k: v for k, v in cfg.items() if k in recognized_fields}

