"""Provides configuration templates and validation for PyTorch learning rate schedulers."""

from copy import deepcopy
from typing import Dict, Any, List, Union, Callable


class Schedulers_Revised_CFGs:
    """Handles scheduler configuration validation and processing."""
    
    # Dictionary mapping scheduler names to their required and optional parameters
    SCHEDULER_CONFIGS = {
        'step_lr': {
            'required': ['step_size'],
            'optional': ['gamma', 'last_epoch', 'verbose'],
            'defaults': {
                'gamma': 0.1,
                'last_epoch': -1,
                'verbose': False
            },
            'types': {
                'step_size': int,
                'gamma': float,
                'last_epoch': int,
                'verbose': bool
            },
            'description': 'Decays learning rate by gamma every step_size epochs'
        },
        
        'multi_step': {
            'required': ['milestones'],
            'optional': ['gamma', 'last_epoch', 'verbose'],
            'defaults': {
                'gamma': 0.1,
                'last_epoch': -1,
                'verbose': False
            },
            'types': {
                'milestones': list,  # List[int]
                'gamma': float,
                'last_epoch': int,
                'verbose': bool
            },
            'description': 'Decays learning rate at specified milestones'
        },
        
        'reduce_on_plateau': {
            'required': ['mode'],
            'optional': ['factor', 'patience', 'threshold', 'threshold_mode',
                        'cooldown', 'min_lr', 'eps', 'verbose'],
            'defaults': {
                'mode': 'min',
                'factor': 0.1,
                'patience': 10,
                'threshold': 1e-4,
                'threshold_mode': 'rel',
                'cooldown': 0,
                'min_lr': 0,
                'eps': 1e-8,
                'verbose': False
            },
            'types': {
                'mode': str,
                'factor': float,
                'patience': int,
                'threshold': float,
                'threshold_mode': str,
                'cooldown': int,
                'min_lr': (float, list),
                'eps': float,
                'verbose': bool
            },
            'description': 'Reduces learning rate when metric stops improving'
        },
        
        'cosine': {
            'required': ['T_max'],
            'optional': ['eta_min', 'last_epoch', 'verbose'],
            'defaults': {
                'T_max': 100,
                'eta_min': 0.,
                'last_epoch': -1,
                'verbose': False
            },
            'types': {
                'T_max': int,
                'eta_min': float,
                'last_epoch': int,
                'verbose': bool
            },
            'description': 'Cosine annealing schedule'
        },
        
        'exponential': {
            'required': ['gamma'],
            'optional': ['last_epoch', 'verbose'],
            'defaults': {
                'last_epoch': -1,
                'verbose': False
            },
            'types': {
                'gamma': float,
                'last_epoch': int,
                'verbose': bool
            },
            'description': 'Exponential learning rate decay'
        },
        
        'cosine_warm_restarts': {
            'required': ['T_0'],
            'optional': ['T_mult', 'eta_min', 'last_epoch', 'verbose'],
            'defaults': {
                'T_mult': 1,
                'eta_min': 0,
                'last_epoch': -1,
                'verbose': False
            },
            'types': {
                'T_0': int,
                'T_mult': int,
                'eta_min': float,
                'last_epoch': int,
                'verbose': bool
            },
            'description': 'Cosine annealing with warm restarts'
        },
        
        'one_cycle': {
            'required': ['max_lr', 'total_steps'],
            'optional': ['epochs', 'steps_per_epoch', 'pct_start', 'anneal_strategy',
                        'cycle_momentum', 'base_momentum', 'max_momentum', 'div_factor',
                        'final_div_factor', 'last_epoch', 'verbose'],
            'defaults': {
                'pct_start': 0.3,
                'anneal_strategy': 'cos',
                'cycle_momentum': True,
                'base_momentum': 0.85,
                'max_momentum': 0.95,
                'div_factor': 25.0,
                'final_div_factor': 1e4,
                'last_epoch': -1,
                'verbose': False
            },
            'types': {
                'max_lr': (float, list),
                'total_steps': int,
                'epochs': int,
                'steps_per_epoch': int,
                'pct_start': float,
                'anneal_strategy': str,
                'cycle_momentum': bool,
                'base_momentum': float,
                'max_momentum': float,
                'div_factor': float,
                'final_div_factor': float,
                'last_epoch': int,
                'verbose': bool
            },
            'description': 'One cycle learning rate policy'
        }
    }

    def __init__(self, scheduler_dict: dict):
        """Initialize with optimizer and scheduler configurations.
        
        Args:
            optimizer_dict: Optimizer configuration dictionary
            scheduler_dict: Scheduler configuration dictionary
        """
        self.scheduler_dict = deepcopy(scheduler_dict)
        self.name = self.scheduler_dict['name']
        return
        
    def get_scheduler_config(self, name: str) -> Dict[str, Any]:
        """Get configuration template for a specific scheduler.
        
        Args:
            name: Name of the scheduler
            
        Returns:
            Dictionary with scheduler configuration template
            
        Raises:
            ValueError: If scheduler is not recognized
        """
        name = name.lower()
        if name not in self.SCHEDULER_CONFIGS:
            raise ValueError(f"Unrecognized scheduler '{name}'. Available: {list(self.SCHEDULER_CONFIGS.keys())}")
        return deepcopy(self.SCHEDULER_CONFIGS[name])
    
    def validate_scheduler_params(self, name: str, params: Dict[str, Any]) -> None:
        """Validate scheduler parameters against required fields and types.
        
        Args:
            name: Scheduler name
            params: Parameters to validate
            
        Raises:
            ValueError: If required parameters are missing or types are incorrect
        """
        config = self.get_scheduler_config(name)
        
        # Check required parameters
        missing = [p for p in config['required'] if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters for {name} scheduler: {missing}")
        
        # Check parameter types
        for param, value in params.items():
            if param in config['types']:
                expected_type = config['types'][param]
                # Handle Union types
                if isinstance(expected_type, tuple):
                    if not any(isinstance(value, t) for t in expected_type):
                        raise TypeError(f"Parameter '{param}' should be one of {expected_type}")
                elif not isinstance(value, expected_type):
                    raise TypeError(f"Parameter '{param}' should be {expected_type}")
    
    def get_revised_dict(self, **cfg) -> Dict[str, Any]:
        """Create a validated scheduler configuration.
        
        Args:
            scheduler_name: Name of the scheduler
            **cfg: Scheduler parameters
            
        Returns:
            Validated scheduler configuration dictionary
        """
        
        scheduler_name = self.name
        
        config = self.get_scheduler_config(scheduler_name)
        
        # Start with defaults
        result = {}
        # result = {'name': scheduler_name}
        result.update(config['defaults'])
        
        # Update with provided values
        result.update({k: v for k, v in cfg.items() 
                      if k in config['required'] or k in config['optional']})
        
        # Validate
        self.validate_scheduler_params(scheduler_name, result)
        
        return result

