import torch
import numpy as np
from copy import deepcopy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model
from _0_general_ML.data_utils.datasets import GTSRB

from ...adversarial_attacks.all_available_adversarial_attacks import FGSM, i_FGSM

from .activation_clustering_official.activation_clustering_helper import get_wrapped_model
from .activation_clustering_official.activation_clustering_official import Activation_Clustering_Official

from ...backdoor_attacks.simple_backdoor import Simple_Backdoor
from .backdoor_defense import Backdoor_Detection_Defense

from utils_.torch_utils import get_data_samples_from_loader, get_outputs, prepare_dataloader_from_numpy



class Multi_Domain_Trojan_Detector(Backdoor_Detection_Defense):
    
    def __init__(self, torch_model: Torch_Model=None, defense_configuration: dict={}, **kwargs):
        
        super().__init__(torch_model=torch_model, defense_configuration=defense_configuration)
        
        return
    
    
    def configure_defense(self, defense_configuration: dict={}):
        
        super().configure_defense(defense_configuration=defense_configuration)
        
        default_configuration = {
            'num_samples': 10,
            'allowed_fps': 0.01
        }
        for key in default_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_configuration[key]
                
        self.num_samples = self.configuration['num_samples']
        self.batch_size = self.torch_model.model_configuration['batch_size']
        
        return
    
    
    def get_distances(self, x, y):
        
        x_adv = self.attack.attack(x, y, epsilon=0.2, targeted=False)
        perturbation_direction = np.sign(x_adv-x)
        
        eps = np.arange(0, 0.51, 0.01)
        distances = 0.3 * np.ones((len(x)))
        for ep in eps[::-1]:
            new_x = x + ep * perturbation_direction
            new_y = np.argmax(get_outputs(self.torch_model.model, prepare_dataloader_from_numpy(new_x, new_x, batch_size=self.batch_size), return_numpy=True), axis=1)
            distances[new_y!=y] = ep
        
        return distances
    
    
    def defend(self, *args, **kwargs):
        
        # get clean samples
        x, y = get_data_samples_from_loader(torch.utils.data.DataLoader(self.torch_model.data.test, batch_size=self.batch_size), return_numpy=True)
        self.num_samples = self.num_samples if self.num_samples<len(x) else len(x)
        random_indices = np.random.choice(len(x), size=self.num_samples, replace=False) 
        x, y = x[random_indices], y[random_indices]
        
        self.attack = FGSM(self.torch_model)
        distances = self.get_distances(x, y)
        
        distances = distances[np.where(distances>0)]
        self.distances = deepcopy(distances)
        self.mean_distances = np.mean(distances)
        self.std_distances = np.std(distances)
        self.t_quantile = np.sort(distances)[min(len(distances)-1, 1+int(len(distances)*(1-0.5*self.configuration['allowed_fps'])))]
        
        if self.num_samples > 30:
            self.threshold_distance = self.t_quantile
        else:
            self.alpha = self.t_quantile * self.mean_distances / (self.std_distances * np.sqrt(len(distances)))
            self.threshold_distance = self.alpha * self.std_distances + self.mean_distances
        
        return
    
    
    def evaluate(self, data_in: Simple_Backdoor, *args, **kwargs):
        
        def get_purified_output(dl: torch.utils.data.DataLoader):
            x, y = get_data_samples_from_loader(dl, return_numpy=True)
            pred_y = get_outputs(self.torch_model.model, dl, return_numpy=True)
            output_random = torch.normal(0, 10, size=pred_y.shape)
            distances = self.get_distances(x, y)
            # vulnerable_indices = ((distances-self.mean_distances) > self.alpha*self.std_distances)
            vulnerable_indices = (distances>self.threshold_distance)
            pred_y[vulnerable_indices] = output_random[vulnerable_indices]
            return pred_y, y, np.mean(np.argmax(pred_y, axis=1)==y)
        
        clean_dataloader = torch.utils.data.DataLoader(data_in.test, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False)
        poisoned_dataloader = torch.utils.data.DataLoader(data_in.poisoned_test, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False)
        
        pred_cy, cy, acc_clean = get_purified_output(clean_dataloader)
        pred_py, py, acc_poisoned = get_purified_output(poisoned_dataloader)
        
        loss_clean, _, _ = self.torch_model.test_shot(clean_dataloader); print()
        loss_poisoned, _, _ = self.torch_model.test_shot(poisoned_dataloader)
        
        return (loss_clean, acc_clean), (loss_poisoned, acc_poisoned)
    
    
    def __purify_model(self, model: torch.nn.Module=None, training_mode: bool=False, **kwargs):
        
        class Purified_Net(torch.nn.Module):
            
            def __init__(local_self, net, training_mode: bool=False):
                
                super().__init__()
                
                assert net is not None, 'model is None. Please pass a model to be purified for STRIP to work.'
                
                local_self.net = net
                
                return
            
            
            def detect(local_self, dataset):
                features_ = self.feature_model(dataset).detach().cpu().numpy()
                t_features_clean = self.activations_clustering_official.decomp_all[self.most_vulnerable_class].transform(features_)
                l_features_clean = self.activations_clustering_official.kmeans_all[self.most_vulnerable_class]['algo'].predict(t_features_clean)
                return l_features_clean!=self.good_label
            
            
            def __call__(local_self, dataset):
                
                output = local_self.net(dataset.to(self.torch_model.device))
                output_classes = output.argmax(1).detach().cpu().numpy().reshape(-1)
                
                positives = local_self.detect(dataset)
                vulnerable_ind = (output_classes==self.most_vulnerable_class)
                vulnerable_ind = vulnerable_ind & positives
                
                output_random = torch.normal(0, 10, size=output.shape).to(output.device)
                output[vulnerable_ind] = output_random[vulnerable_ind].clone()
                
                return output
            
            
        return Purified_Net(model, training_mode=training_mode) if self.backdoor_found else model
    
    