import torch

from captum._utils.common import _format_output
from captum._utils.gradient import _forward_layer_eval


from .torch_model import Torch_Model



torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def activations_with_grad(inputs, model, last_layer, attribute_to_layer_input=True, grad_enabled=False):
    """
    This function has been directly imported and used from https://github.com/YanNeu/spurious_imagenet.
    """
    
    if len(inputs.shape) == 3:
        inputs = inputs[None, :]

    layer_eval = _forward_layer_eval(
        model,
        inputs,
        last_layer,
        None,
        attribute_to_layer_input=attribute_to_layer_input,
        grad_enabled=grad_enabled
    )
    
    if isinstance(last_layer, torch.nn.Module):
        return _format_output(len(layer_eval) > 1, layer_eval)
    else:
        return [
            _format_output(len(single_layer_eval) > 1, single_layer_eval)
            for single_layer_eval in layer_eval
        ]


class Feature_Activations(torch.nn.Module):
    
    def __init__(self, model_, last_layer_, target_class_, mode='default', device: str=torch_device):
        
        super().__init__()
        
        self.model = model_
        self.last_layer = last_layer_
        self.target_class = target_class_
        
        self.mode = mode
        self.device = device
        
        try: self.model.eval()
        except: print('Could not make the model into eval mode. Please do it explicitly where possible.')
        
        return
    
    
    def default_forward(self, X):
        return activations_with_grad(X, self.model, self.last_layer, grad_enabled=True)  * self.last_layer.weight[self.target_class]
    def logit_forward(self, X):
        return activations_with_grad(X, self.model, self.last_layer, grad_enabled=True)  @ self.last_layer.weight[self.target_class]
    def forward_until_last_layer(self, X):
        return activations_with_grad(X, self.model, self.last_layer, grad_enabled=True)
    def classification_forward(self, X):
        return self.model(X)
    def forward(self, X):
        if self.mode == 'logit': return self.logit_forward(X)
        elif self.mode == 'only_activations': return self.forward_until_last_layer(X)
        elif self.mode == 'classification': return self.classification_forward(X)
        return self.default_forward(X)
    
    
