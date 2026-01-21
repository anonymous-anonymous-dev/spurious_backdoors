import torch
import numpy as np
from copy import deepcopy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model
from _0_general_ML.data_utils.datasets import GTSRB

from .activation_clustering_official.activation_clustering_helper import get_wrapped_model
from .activation_clustering_official.activation_clustering_official import Activation_Clustering_Official

from ...backdoor_attacks.simple_backdoor import Simple_Backdoor
from .backdoor_defense import Backdoor_Detection_Defense

from utils_.torch_utils import get_data_samples_from_loader, get_outputs, prepare_dataloader_from_numpy



class Activation_Clustering(Backdoor_Detection_Defense):
    
    def __init__(self, torch_model: Torch_Model=None, defense_configuration: dict={}, **kwargs):
        
        super().__init__(torch_model=torch_model, defense_configuration=defense_configuration)
        
        return
    
    
    def configure_defense(self, defense_configuration: dict={}):
        
        super().configure_defense(defense_configuration)
        
        default_configuration = {
            # 'number_of_clean_samples_for_computing_threshold': 100,
            'unofficial_threshold': 2,
            'tolerance': -0.02
        }
        for key in default_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_configuration[key]
        
        self.activations_clustering_official = Activation_Clustering_Official(self.torch_model)
        self.feature_model = get_wrapped_model(self.torch_model, target_class=0)
        self.available_data = self.torch_model.data
        
        # self.feature_model.mode = 'default'
        
        return
    
    
    def defend_mr(self, epochs: int=10, *args, **kwargs):
        self.torch_model.train(epochs=epochs)
        self.defend()
        return
    
    
    def defend(self, *args, mode: str='official', compute_custom_threshold: bool=True, **kwargs):
        
        # ood_data = GTSRB(preferred_size=data_in.preferred_size, data_means=data_in.data_means, data_stds=data_in.data_stds)
        
        clean_dataloader = torch.utils.data.DataLoader(self.available_data.test, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False)
        train_dataloader = torch.utils.data.DataLoader(
            self.available_data.train if self.available_data.train is not None else self.available_data.test, 
            batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False
        )
        
        features_clean = get_outputs(self.feature_model, clean_dataloader, return_numpy=True)
        if compute_custom_threshold:
            for t, target_class in enumerate(range(self.torch_model.data.num_classes)):
                pre_str = f'\rTesting for threshold {target_class}, which is {t+1}/{self.torch_model.data.num_classes}. '
                print(pre_str, end='')
                # create lists of clean and poison indices
                _z_indices = np.where(np.array(self.available_data.test.targets)==target_class)[0]
                self.activations_clustering_official.activation_clustering_defense_custom(features_clean[_z_indices], target_class=target_class, verbose=False)
            self.activations_clustering_official.threshold = np.max(self.activations_clustering_official.all_scores) * (1+self.configuration['tolerance'])
        else:
            mode = 'unofficial'
        
        features_train = get_outputs(self.feature_model, train_dataloader, return_numpy=True)
        for t, target_class in enumerate(range(self.torch_model.data.num_classes)):
            pre_str = f'\rAnalyzing class {target_class}, which is {t+1}/{self.torch_model.data.num_classes}. '
            print(pre_str, end='')
            # create lists of clean and poison indices
            _z_indices = np.where(np.array(self.available_data.train.targets)==target_class)[0]
            self.activations_clustering_official.activation_clustering_defense_custom(features_train[_z_indices], target_class=target_class)    
            # if target_class==0:
            #     import matplotlib.pyplot as plt
            #     dcomp = self.activations_clustering_official.decomp_all[target_class]
            #     all_features_transformed = dcomp.transform(features_train[_z_indices])
            #     feature_labels = self.activations_clustering_official.kmeans_all[target_class]['algo'].predict(all_features_transformed)
            #     plt.scatter(all_features_transformed[feature_labels==0,0], all_features_transformed[feature_labels==0,1], c='blue', alpha=0.5)
            #     plt.scatter(all_features_transformed[feature_labels==1,0], all_features_transformed[feature_labels==1,1], c='red', alpha=0.5)
            #     print('Silhouette Score:', self.activations_clustering_official.all_scores[target_class], 'Threshold:', self.activations_clustering_official.threshold)
        print()
        
        # process scores (if mode=='unofficial') for generating backdoor probabilities
        self.process_scores(self.activations_clustering_official.all_scores, mode=mode)
        self.highest_score, self.most_vulnerable_class = np.max(self.processed_scores), np.argmax(self.processed_scores)
        if self.highest_score > self.threshold:
            self.backdoor_found = True
            t_features_clean = self.activations_clustering_official.decomp_all[self.most_vulnerable_class].transform(features_clean[np.where(np.array(self.available_data.test.targets)==self.most_vulnerable_class)[0]])
            l_features_clean = self.activations_clustering_official.kmeans_all[self.most_vulnerable_class]['algo'].predict(t_features_clean)
            self.good_label = 1 if np.mean(l_features_clean)>0.5 else 0
        else:
            self.backdoor_found = False
            self.good_label = -1
        
        # Now testing
        altered_model = Torch_Model(self.available_data, self.torch_model.model_configuration, path='')
        altered_model.model.load_state_dict(self.torch_model.model.state_dict())
        altered_model.model = self.purify_model(altered_model.model)
        
        # self.final_model = deepcopy(altered_model)
        self.final_model = Torch_Model(altered_model.data, altered_model.model_configuration, path=altered_model.path)
        self.final_model.model = deepcopy(altered_model.model)
        
        # loss_clean, acc_clean = altered_model.test_shot(clean_dataloader)
        # print()
        # loss_poisoned, acc_poisoned = altered_model.test_shot(poisoned_dataloader)
        
        # return (loss_clean, acc_clean), (loss_poisoned, acc_poisoned)
        return
    
    
    def process_scores(self, all_scores, mode='official'):
        
        if mode == 'unofficial':
            self.threshold = self.configuration['unofficial_threshold']
            self.processed_scores = (all_scores - np.median(all_scores)) / np.std(all_scores)
        else:
            self.threshold = self.activations_clustering_official.threshold
            self.processed_scores = self.activations_clustering_official.all_scores
        
        return
    
    
    def print_report(self, data_in: Simple_Backdoor):
        
        if self.backdoor_found:
            self.highest_score, self.most_vulnerable_class = np.max(self.processed_scores), np.argmax(self.processed_scores)
            if self.most_vulnerable_class == data_in.targets[0]:
                _z_indices = np.where(np.array(data_in.train.targets)==data_in.targets[0])[0]
                clean_indices, poison_indices = [], []
                for i, k in enumerate(_z_indices):
                    if k in data_in.train.poison_indices:
                        poison_indices.append(i)
                    else:
                        clean_indices.append(i)
                        
                if len(poison_indices)>0:
                    self.activations_clustering_official.cleaning_model(clean_indices, poison_indices)
                        
            vulnerable_classes = np.where(self.processed_scores>self.threshold)[0].reshape(-1)
            self.most_vulnerable_classes = np.where(self.processed_scores==np.max(self.processed_scores))[0].reshape(-1)
            if (len(vulnerable_classes)>0):
                print(f'\nAttack detected ! The most vulnerable class is: {self.most_vulnerable_class} with score {np.max(self.processed_scores):.5f}.')
                print(f'Other vulnerable classes are: {', '.join([f'\n{class_}, \t{self.processed_scores[class_]:.5f}' for class_ in vulnerable_classes])}\n.')
        else:
            print('\nNo backdoor class found.')
        
        return
    
    
    def prepare_threshold(self, features_clean, num_samples: int=100):
        scores = self.activations_clustering_official.activation_clustering_defense_custom(features_clean)
        self.activations_clustering_official.threshold = np.max(scores)
        return
    
    
    def purify_model(self, model: torch.nn.Module=None, training_mode: bool=False, **kwargs):
        
        class Purified_Net(torch.nn.Module):
            
            def __init__(local_self, net, training_mode: bool=False):
                
                super().__init__()
                
                assert net is not None, 'model is None. Please pass a model to be purified for STRIP to work.'
                
                local_self.net = net
                local_self.training_mode = training_mode
                
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
    
    