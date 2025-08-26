import torch


from .torch_model import Torch_Model



class Hooker:
    
    def __init__(self, layer: torch.nn.Module, target_class: int=0):
        self.layer = layer
        self.target_class = target_class
        self.value = None
        return
    
    def h(self, model, input, output):
        self.value = output.detach()
        return 
    
    def weighted_h(self, model, input, output):
        self.value = (input[0] * self.layer.weight[0]).detach()
        return
    
    
class Dependable_Feature_Activations(torch.nn.Module):
    
    def __init__(
        self, 
        model: Torch_Model, layer_numbers: int=[-1], 
        get_weighted_activations_for_the_last_layer: bool=False,
        **kwargs
    ):
        
        super().__init__()
        
        self.model = model.model
        self.layers = [model.get_children(model.model)[layer_number] for layer_number in layer_numbers]
        self.hook_fns = {layer_number: Hooker(self.layers[l]) for l, layer_number in enumerate(layer_numbers)}
        self.all_hooks = [
            layer.register_forward_hook(self.hook_fns[layer_numbers[l]].weighted_h if (layer_numbers[l]==-1)&(get_weighted_activations_for_the_last_layer) \
                else self.hook_fns[layer_numbers[l]].h) 
            for l, layer in enumerate(self.layers)
        ]
        
        return
    
    
    def forward(self, x):
        _ = self.model(x)
        return torch.cat([self.hook_fns[k].value.view(len(x), -1) for k in self.hook_fns.keys()], dim=1)
    
    