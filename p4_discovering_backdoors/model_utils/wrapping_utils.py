import torch


from _0_general_ML.model_utils.torch_model import Torch_Model

from .robust_torch_model import Robustify_Model
# from .._paper_implementations_.spurious_imagenet_utils import TemperatureWrapper, ImageNetWrapper

# from .._imported_implementations_.spurious_imagenet.neural_pca.adversarial_attacks.act_apgd import ActivationAPGDAttack
from .._imported_implementations_.spurious_imagenet.utils.temperature_wrapper import TemperatureWrapper
# from .._imported_implementations_.spurious_imagenet.utils.model_normalization import ImageNetWrapper

from .latentify_model import Latentify_Model
from .feature_activations import Feature_Activations



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NormalizationWrapper(torch.nn.Module):
    def __init__(self, model, mean, std, device=torch_device):
        super().__init__()

        mean = torch.tensor(mean).to(device)
        std = torch.tensor(std).to(device)

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
        self.normalized_model = True
                
        return

    def forward(self, x, *args, **kwargs):
        # print(self.mean.shape, x.shape)
        x_normalized = (x - self.mean)/self.std
        return self.model(x_normalized, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict()


def get_wrapped_model(
    model_in: Torch_Model, target_class: int,
    robustification_wrap: bool=False, normalization_wrap: bool=False, temperature_wrap: bool=False, latentification_wrap: bool=False,
    return_last_layer: bool=False
):
    
    _model = model_in.model
    
    if model_in.data.data_name == 'kaggle_imagenet':
        _last_layer = list(_model.backbone.children())[-1] if 'vit' not in model_in.model_configuration['model_architecture'] else list(list(_model.backbone.children())[-1].children())[-1]
    else:
        # TODO: We should later bring consistency to these variable names.
        try: _last_layer = _model.fl2 # for cifar10 and gtsrb
        except: 
            try: _last_layer = _model.fc2 # for mnist architecture
            except: 
                try: _last_layer = _model.fc # for vit architecture
                except: _last_layer = _model.linear
    # print(_last_layer)
    # assert _last_layer.out_features == 1000
    
    if robustification_wrap:
        _model = Robustify_Model(model_in.model, autoencoder_configuration={'noise_mag': 0.3, 'n_iter': 1, 'device': model_in.device})
    if normalization_wrap:
        print('Model being normalized.')
        _model = NormalizationWrapper(_model, model_in.data.data_means, model_in.data.data_stds, device=model_in.device)
    if temperature_wrap:
        print('Model being temperature wrapped.')
        _model = TemperatureWrapper(_model, T=0.7155761122703552).to(model_in.device)
        # _model = TemperatureWrapper(_model, T=0.7155761122703552).to(model_in.device)
    if latentification_wrap:
        _model = Latentify_Model(_model)
    _model = Feature_Activations(_model, _last_layer, target_class_=target_class)
    
    if return_last_layer:
        return _model, _last_layer
    
    return _model

