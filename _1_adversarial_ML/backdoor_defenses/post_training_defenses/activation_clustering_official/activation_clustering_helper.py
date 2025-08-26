import torch


from _0_general_ML.model_utils.torch_model import Torch_Model
from _0_general_ML.model_utils.activations_wrapper import Feature_Activations



def get_wrapped_model(model_in: Torch_Model, target_class: int=0):
    
    _model = model_in.model
    
    if model_in.data.data_name == 'kaggle_imagenet':
        _last_layer = list(_model.backbone_model.children())[-1] if 'vit' not in model_in.model_configuration['model_architecture'] else list(list(_model.backbone_model.children())[-1].children())[-1]
    else:
        # TODO: We should later bring consistency to these variable names.
        try: _last_layer = _model.fl2 # for cifar10 and gtsrb
        except: _last_layer = _model.fc2 # for mnist architecture
    print(_last_layer)
    # assert _last_layer.out_features == 1000
    
    _model = Feature_Activations(_model, _last_layer, target_class_=target_class, mode='only_activations')
    
    return _model

