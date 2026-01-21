import torch
import numpy as np
from copy import deepcopy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from .activation_clustering_official.activation_clustering_helper import get_wrapped_model
# from .activation_clustering_official.activation_clustering_official import Activation_Clustering_Official
# from .spectral_signatures_official.spectral_signatures_official import Spectral_Signatures_Official
from .spectral_signatures_official.spectral_official import Spectral_Signatures_Official

from ...backdoor_attacks.simple_backdoor import Simple_Backdoor
from .backdoor_defense import Backdoor_Detection_Defense

from utils_.torch_utils import get_data_samples_from_loader, get_outputs, prepare_dataloader_from_numpy



class Spectral_Clustering(Backdoor_Detection_Defense):
    
    def __init__(self, torch_model: Torch_Model=None, defense_configuration: dict={}, **kwargs):
        
        super().__init__(torch_model=torch_model, defense_configuration=defense_configuration)
        
        return
    
    
    def configure_defense(self, defense_configuration: dict={}):
        
        super().configure_defense(defense_configuration)
        
        default_configuration = {
            # 'number_of_clean_samples_for_computing_threshold': 100,
            'unofficial_threshold': 2,
            'tolerance': -0.01
        }
        for key in default_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_configuration[key]
        
        self.official_ss = Spectral_Signatures_Official(self.torch_model)
        self.feature_model = get_wrapped_model(self.torch_model, target_class=0)
        # self.feature_model.mode = 'default'
        
        return
    
    
    def defend_mr(self, epochs: int=10, *args, **kwargs):
        self.torch_model.train(epochs=epochs)
        self.defend()
        return
    
    
    def defend(self, *args, **kwargs):
        
        data_in = self.torch_model.data
        train_dataloader = torch.utils.data.DataLoader(
            data_in.train if data_in.train is not None else data_in.test, 
            batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False
        )
        features_train = get_outputs(self.feature_model, train_dataloader, return_numpy=True)
        
        for t, target_class in enumerate(range(self.torch_model.data.num_classes)):
            pre_str = f'\rAnalyzing class {target_class}, which is {t+1}/{self.torch_model.data.num_classes}. '
            print(pre_str, end='')
            # create lists of clean and poison indices
            _z_indices = np.where(np.array(data_in.train.targets)==target_class)[0]
            self.official_ss.prepare_defense(features_train[_z_indices], target_class)
            
        # Now testing
        altered_model = Torch_Model(data_in, self.torch_model.model_configuration, path='')
        altered_model.model.load_state_dict(self.torch_model.model.state_dict())
        altered_model.model = self.purify_model(altered_model.model)
        
        # self.final_model = deepcopy(altered_model)
        self.final_model = Torch_Model(altered_model.data, altered_model.model_configuration, path=altered_model.path)
        self.final_model.model = deepcopy(altered_model.model)
        
        # clean_dataloader = torch.utils.data.DataLoader(data_in.test, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False)
        # poisoned_dataloader = torch.utils.data.DataLoader(data_in.poisoned_test, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False)
        # loss_clean, acc_clean = altered_model.test_shot(clean_dataloader); print()
        # loss_poisoned, acc_poisoned = altered_model.test_shot(poisoned_dataloader); print()
        
        # return (loss_clean, acc_clean), (loss_poisoned, acc_poisoned)
        return
    
    
    def purify_model(self, model: torch.nn.Module=None, training_mode: bool=False, **kwargs):
        
        class Purified_Net(torch.nn.Module):
            
            def __init__(local_self, net, training_mode: bool=False):
                
                super().__init__()
                
                assert net is not None, 'model is None. Please pass a model to be purified for STRIP to work.'
                
                local_self.net = net
                local_self.training_mode = training_mode
                
                return
            
            
            def detect(local_self, dataset, output_classes):
                features_ = self.feature_model(dataset).detach().cpu().numpy()
                metric = self.official_ss.detect_with_ss_metric(features_, output_classes)
                return metric
            
            
            def __call__(local_self, dataset):
                
                output = local_self.net(dataset.to(self.torch_model.device))
                output_classes = output.argmax(1).detach().cpu().numpy().reshape(-1)
                
                positives = local_self.detect(dataset, output_classes)
                
                output_random = torch.normal(0, 10, size=output.shape).to(output.device)
                output[positives] = output_random[positives].clone()
                
                return output
            
            
        return Purified_Net(model)
    
    