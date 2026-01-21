import torch
from copy import deepcopy
from typing import Optional, Dict, Any, Union, Callable
import warnings
from torch.optim import lr_scheduler


from .optimizers_recognized_fields import Revised_CFGs
from .schedulers_recognized_fields import Schedulers_Revised_CFGs



class Torch_Optimizer:
    
    def __init__(
        self,
        parameters,
        optimizer_dict: Optional[Dict[str, Any]] = None,
        scheduler_dict: Optional[Dict[str, Any]] = None
    ):
        """Initialize optimizer and scheduler configurations.
        
        Args:
            parameters: Model parameters to optimize
            optimizer_dict: Configuration for optimizer. Example:
                {
                    'name': 'adam',  # Choose from: adam, adamw, sgd, rmsprop, adagrad, adadelta, lbfgs
                    'lr': 1e-3,
                    'weight_decay': 0.01,
                    # Additional args specific to each optimizer:
                    # SGD: momentum, nesterov
                    # Adam/AdamW: betas, eps, amsgrad
                    # RMSprop: alpha, momentum, centered
                    # Adagrad: lr_decay, initial_accumulator_value
                    # Adadelta: rho, eps
                    # LBFGS: max_iter, max_eval, tolerance_grad, tolerance_change, history_size
                }
            scheduler_dict: Configuration for scheduler. Example:
                {
                    'name': 'step',  # Choose from: step, multistep, exponential, cosine, reduce_on_plateau, one_cycle, cosine_warm
                    'auto_step': True,  # Whether to step automatically after optimizer.step()
                    # Additional args specific to each scheduler:
                    # StepLR: step_size, gamma
                    # MultiStepLR: milestones, gamma
                    # ExponentialLR: gamma
                    # CosineAnnealingLR: T_max, eta_min
                    # ReduceLROnPlateau: mode, factor, patience, threshold, cooldown
                    # OneCycleLR: max_lr, total_steps, pct_start
                    # CosineAnnealingWarmRestarts: T_0, T_mult
                }
        """
        self.parameters = parameters
        
        # Set up optimizer configuration
        default_optimizer_dict = {
            'name': 'adam',
            'lr': 1e-3,
            'weight_decay': 0.0
        }
        self.optimizer_dict = default_optimizer_dict.copy()
        if optimizer_dict:
            self.optimizer_dict.update(optimizer_dict)
            
        # Set up scheduler configuration
        default_scheduler_dict = {
            'name': None,  # No scheduler by default
            'auto_step': True
        }
        self.scheduler_dict = default_scheduler_dict.copy()
        if scheduler_dict:
            self.scheduler_dict.update(scheduler_dict)
            
        # Initialize optimizer and scheduler as None
        self.optim = None
        self.scheduler = None
        self._scheduler_auto_step = True
        
        self.optim = None
        self.scheduler = None
        # whether to call scheduler.step() automatically after optimizer.step()
        self._scheduler_auto_step = True
        
        self.optim_revised_cfgs = Revised_CFGs(self.optimizer_dict)
        self.scheduler_revised_cfgs = Schedulers_Revised_CFGs(self.scheduler_dict)
        self.prepare_everything()
        
        return
    
    
    def zero_grad(self):
        return self.optim.zero_grad()
    def state_dict(self):
        return self.optim.state_dict()
    def load_state_dict(self, *args, **kwargs):
        return self.optim.load_state_dict(*args, **kwargs)
    
    
    def step(self):
        # call optimizer step
        if self.optim is None:
            raise RuntimeError('Optimizer has not been created. Call return_optimizer(...) first.')

        self.optim.step()

        # call scheduler step automatically if configured
        if self.scheduler is not None and self._scheduler_auto_step:
            try:
                # Some schedulers (ReduceLROnPlateau) expect a metric and should not be auto-stepped here
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # do not auto-step ReduceLROnPlateau because it requires a metric
                    pass
                else:
                    self.scheduler.step()
            except TypeError:
                # If scheduler.step() signature differs, skip automatic stepping and warn
                warnings.warn('Scheduler.step() could not be called automatically; scheduler requires different arguments.', UserWarning)

        return
    
    
    def prepare_everything(self) -> None:
        """Prepare both optimizer and scheduler based on their configurations."""
        self.prepare_optimizer()
        self.prepare_scheduler()
        return
    
    
    def prepare_optimizer(self) -> None:
        """Create and store optimizer based on optimizer_dict configuration."""
        if not hasattr(self, 'parameters'):
            raise ValueError("No parameters set. Initialize with model parameters first.")
            
        name = self.optimizer_dict.get('name', 'adam').lower()
        cfg = {k: v for k, v in self.optimizer_dict.items() if k != 'name'}
        
        optimizers = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
            'adagrad': torch.optim.Adagrad,
            'adadelta': torch.optim.Adadelta,
            'lbfgs': torch.optim.LBFGS
        }
        
        if name not in optimizers:
            raise ValueError(f"Unsupported optimizer {name}. Choose from: {list(optimizers.keys())}")
            
        # Handle specific optimizer requirements
        if name == 'lbfgs':
            if 'max_iter' not in cfg:
                cfg['max_iter'] = 20  # Default max iterations for LBFGS
        elif name == 'sgd' and 'momentum' not in cfg:
            cfg['momentum'] = 0.0  # Default momentum for SGD
        elif name == 'cosine':
            cfg['T_max'] = 100
            
        try:
            rev_cfg = self.optim_revised_cfgs.get_revised_dict(**cfg)
            # print(rev_cfg)
            self.optim = optimizers[name](self.parameters, **rev_cfg)
        except TypeError as e:
            raise TypeError(f"Error creating {name} optimizer: {str(e)}")
        
        return
    
            
    def prepare_scheduler(self) -> None:
        """Create and store scheduler based on scheduler_dict configuration."""
        if self.optim is None:
            raise RuntimeError("Optimizer must be prepared before scheduler.")
            
        name = self.scheduler_dict.get('name')
        if not name:  # No scheduler requested
            self.scheduler = None
            return
            
        cfg = {k: v for k, v in self.scheduler_dict.items() if k not in ['name', 'auto_step']}
        self._scheduler_auto_step = self.scheduler_dict.get('auto_step', True)
        
        schedulers = {
            'step': lr_scheduler.StepLR,
            'multistep': lr_scheduler.MultiStepLR,
            'exponential': lr_scheduler.ExponentialLR,
            'cosine': lr_scheduler.CosineAnnealingLR,
            'reduce_on_plateau': lr_scheduler.ReduceLROnPlateau,
            'one_cycle': lr_scheduler.OneCycleLR,
            'cosine_warm': lr_scheduler.CosineAnnealingWarmRestarts,
        }
        
        name = name.lower()
        if name not in schedulers:
            raise ValueError(f"Unsupported scheduler {name}. Choose from: {list(schedulers.keys())}")
            
        # Handle specific scheduler requirements
        if name == 'reduce_on_plateau':
            if 'mode' not in cfg:
                cfg['mode'] = 'min'  # Default mode
            self._scheduler_auto_step = False  # Requires metric value
        elif name == 'one_cycle':
            if 'total_steps' not in cfg:
                raise ValueError("OneCycleLR requires 'total_steps' parameter")
                
        try:
            rev_cfg = self.scheduler_revised_cfgs.get_revised_dict(**cfg)
            # print(rev_cfg)
            self.scheduler = schedulers[name](self.optim, **rev_cfg)
        except TypeError as e:
            raise TypeError(f"Error creating {name} scheduler: {str(e)}")
        
        return
    
    
    