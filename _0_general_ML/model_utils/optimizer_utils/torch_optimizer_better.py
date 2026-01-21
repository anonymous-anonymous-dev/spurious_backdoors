import torch
import math
from copy import deepcopy
from typing import Optional, Dict, Any, Union, Callable, Iterable
import warnings
from torch.optim import lr_scheduler


class Torch_Optimizer(torch.optim.Optimizer):
    
    def __init__(
        self,
        params,
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
        # Set up optimizer configuration
        default_optimizer_dict = {
            'name': 'adam',
            'lr': 1e-3,
            'weight_decay': 0.0,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
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
            
        # Initialize base Optimizer
        kwargs = {k: v for k, v in self.optimizer_dict.items() 
                 if k not in ['name', 'lr', 'weight_decay']}
        super().__init__(
            params,
            {
                'lr': self.optimizer_dict['lr'],
                'weight_decay': self.optimizer_dict['weight_decay'],
                **kwargs
            }
        )
        
        self.scheduler = None
        self._scheduler_auto_step = True
        
        self.prepare_scheduler()
        
        return
    
    
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    
            name = self.optimizer_dict['name'].lower()
            
            if name == 'adam' or name == 'adamw':
                beta1, beta2 = group['betas']
                
                # Update step using Adam
                self._adam_step(params_with_grad, d_p_list, beta1, beta2, group['eps'],
                              group['weight_decay'], group['lr'])
                              
            elif name == 'sgd':
                momentum = group.get('momentum', 0)
                dampening = group.get('dampening', 0)
                nesterov = group.get('nesterov', False)
                
                # Update step using SGD
                self._sgd_step(params_with_grad, d_p_list, momentum_buffer_list,
                             weight_decay=group['weight_decay'],
                             momentum=momentum, lr=group['lr'],
                             dampening=dampening, nesterov=nesterov)
            
            # Add other optimizers as needed...

        # Handle scheduler step if configured
        if self.scheduler is not None and self._scheduler_auto_step:
            try:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Skip auto-stepping ReduceLROnPlateau
                    pass
                else:
                    self.scheduler.step()
            except TypeError:
                warnings.warn('Scheduler.step() requires different arguments. Skipping auto-step.', UserWarning)

        return loss

    @torch.no_grad()
    def _adam_step(self, params, grads, beta1, beta2, eps, weight_decay, lr):
        for i, param in enumerate(params):
            grad = grads[i]
            
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)
                
            # State initialization
            state = self.state[param]
            
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            
            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            
            step_size = lr / bias_correction1
            
            param.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def _sgd_step(self, params, d_p_list, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov):
        for i, param in enumerate(params):
            d_p = d_p_list[i]
            
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)
                
            if momentum != 0:
                buf = self.state[param].get('momentum_buffer', None)
                
                if buf is None:
                    buf = torch.clone(d_p).detach()
                    self.state[param]['momentum_buffer'] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
                    
            param.add_(d_p, alpha=-lr)
    
    
    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`.
        
        Contains two entries:
        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups
        """
        return {
            'state': self.state,
            'param_groups': self.param_groups,
            'optimizer_dict': self.optimizer_dict,
            'scheduler_dict': self.scheduler_dict,
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the optimizer state.
        
        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # Load optimizer state
        self.state.clear()
        self.state.update(state_dict['state'])
        
        # Load param_groups
        param_groups_copy = [dict(p.items()) for p in state_dict['param_groups']]
        self.param_groups.clear()
        self.param_groups.extend(param_groups_copy)
        
        # Load optimizer and scheduler configurations
        self.optimizer_dict = state_dict['optimizer_dict']
        self.scheduler_dict = state_dict['scheduler_dict']
        
        # Recreate scheduler if needed
        if state_dict['scheduler_state'] is not None:
            self.prepare_scheduler()
            self.scheduler.load_state_dict(state_dict['scheduler_state'])
    
            
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
            self.scheduler = schedulers[name](self.optim, **cfg)
        except TypeError as e:
            raise TypeError(f"Error creating {name} scheduler: {str(e)}")
        
        return
    
    
    