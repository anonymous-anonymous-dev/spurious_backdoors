import torch
import numpy as np
from termcolor import colored


from .torch_model import Torch_Model



class Hooker:
    
    def __init__(self, layer: torch.nn.Module, target_class: int=0, detach_output: bool=True):
        self.layer = layer
        self.target_class = target_class
        self.value = None
        self.detach_output = detach_output
        return
    
    def h(self, model, input, output):
        self.value = output.detach() if self.detach_output else output
        return
    
    def weighted_h(self, model, input, output):
        
        # if layer does not have weights (like ReLU), then weighted activations are not possible. so return normal activations
        if not hasattr(self.layer, 'weight'):
            # print(f'Layer does not have weights, returning normal activations.')
            self.h(model, input, output)
            return
        
        # print(f'Outputing weighted.')
        if self.target_class is not None:
            value = (input[0] * self.layer.weight[self.target_class])
            self.value = value.detach() if self.detach_output else value
        else:
            self.h(model, input, output)
        return
    
    
class Dependable_Feature_Activations(torch.nn.Module):
    
    def __init__(
        self, 
        model: Torch_Model, layer_numbers: int=[-1], target_class: int=0,
        get_weighted_activations_for_the_last_layer: bool=False,
        output_type: str=None, detach_output: bool=True,
        verbose: int=0,
        **kwargs
    ):
        
        super().__init__()
        
        self.model = model.model
        self.target_class = target_class
        self.output_type = 'flattened' if output_type is None else output_type
        self.detach_output = detach_output
        self.device = model.device
        
        self.get_weighted_activations_for_the_last_layer = get_weighted_activations_for_the_last_layer
        
        self.layers = [model.get_children(model.model)[layer_number] for layer_number in layer_numbers]
        
        last_layer = -1
        for l, layer in enumerate(self.layers):
            last_layer = l if hasattr(layer, 'weight') else last_layer
        last_layer = len(self.layers)-1 if last_layer==-1 else last_layer
        self.last_layer = last_layer
        # print(colored(f'[Model Activations Wrapper] layer_numbers: {layer_numbers} and last layer is {last_layer}', 'red'))
        
        self.hook_fns = {layer_number: Hooker(self.layers[l], target_class=target_class, detach_output=detach_output) for l, layer_number in enumerate(layer_numbers)}
        self.all_hooks = [
            layer.register_forward_hook(self.hook_fns[layer_numbers[l]].weighted_h if (l==last_layer)&(get_weighted_activations_for_the_last_layer) \
                else self.hook_fns[layer_numbers[l]].h) 
            for l, layer in enumerate(self.layers)
        ]
        # self.all_hooks = [
        #     layer.register_full_backward_hook(self.hook_fns[layer_numbers[l]].weighted_h if (layer_numbers[l]==-1)&(get_weighted_activations_for_the_last_layer) \
        #         else self.hook_fns[layer_numbers[l]].h) 
        #     for l, layer in enumerate(self.layers)
        # ]
        
        self.verbose = verbose
        
        return
    
    
    def forward(self, x):
        if self.output_type == 'dict':
            return self.dict_forward(x)
        return self.flattened_forward(x)
    
    
    def dict_forward(self, x):
        probs = self.model(x)
        return {k: self.hook_fns[k].value.view(len(x), -1) if self.hook_fns[k].value is not None else probs.clone().view(len(probs), -1) for k in self.hook_fns.keys()}
    
    
    def flattened_forward(self, x):
        _ = self.model(x)
        return torch.cat([self.hook_fns[k].value.view(len(x), -1) for k in self.hook_fns.keys()], dim=1)
    
    