import numpy as np

import torch
import torchvision
import copy

from termcolor import colored

from sklearn.cluster import SpectralClustering, KMeans, HDBSCAN
from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_defenses.post_training_defenses.backdoor_defense import Backdoor_Detection_Defense
from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor

from ..attacks.activation_based_pgd import Activation_Based_PGD

from ..model_utils.wrapping_utils import get_wrapped_model

from utils_.pca import PCA_Loss, General_PCA, PCA_of_SKLEARN, PCA_of_NPCA, Sparse_PCA_of_SKLEARN, PCA_SKLEARN_MEDIAN
from utils_.torch_utils import get_outputs, get_data_samples_from_loader, evaluate_on_numpy_arrays, prepare_dataloader_from_numpy
from utils_.general_utils import normalize



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


all_pca_classes = {
    'pca_npca': PCA_of_NPCA,
    'pca_sklearn': PCA_of_SKLEARN,
    'sparse_pca_sklearn': Sparse_PCA_of_SKLEARN,
    'pca_median': PCA_SKLEARN_MEDIAN
}


class NPCA_Paper(Backdoor_Detection_Defense):
    
    def __init__(
        self, 
        data: Torch_Dataset=None, 
        torch_model: Torch_Model=None, 
        defense_configuration: dict={},
        # target_class: int=0, non_target_class: int=None,
        verbose: bool=True,
        **kwargs
    ):
        
        self.defense_name = 'NPCA'
        
        self.data = data # copy.deepcopy(data)
        self.data_loader = None
        self.data_loader_original = None
        self.original_model = torch_model
        self.model = None
        self.configuration = None
        self.verbose = verbose
        
        self.pca_classes = all_pca_classes
        self.batch_size = torch_model.model_configuration['batch_size']
        
        super().__init__(torch_model=torch_model, defense_configuration=defense_configuration)
        
        return
    
    
    def show_features_and_plots(self, *args, **kwargs):
        len_features = self._pca.transform(self.ac_.detach().cpu().numpy()).shape[-1]
        features_to_show = min(len_features, 10) // 2
        show_features = list(np.arange(features_to_show))+list(len_features-np.arange(features_to_show, 0, -1))
        self.plot_components(show_features=show_features, num_images_for_each_feature=10, use_original_dataloader=True);
        plt.show()
        return
    
    
    def configure_defense(
        self, torch_model: Torch_Model=None, data_loader: torch.utils.data.DataLoader=None, defense_configuration: dict={},
        **kwargs
    ):
        
        # update data loader and torch model
        self.data_loader = data_loader if data_loader is not None else self.data_loader
        self.original_model = torch_model if torch_model is not None else self.original_model
        # defense_configuration = defense_configuration if len(list(defense_configuration.keys()))>0 else self.configuration
        
        # update configuration
        default_configuration = {
            'target_class': 0,
            'non_target_class': None,
            'normalization_wrap': False,
            'temperature_wrap': False,
            'pca_type': 'pca_sklearn',
            'n_components': None,
        }
        self.configuration = default_configuration if self.configuration is None else self.configuration
        for key in defense_configuration.keys():
            self.configuration[key] = defense_configuration[key]
        self.print_out(self.configuration)
        
        # this transformation will be used if normalization_wrapper is on.
        self.redefined_test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.data.preferred_size), 
                # torchvision.transforms.Normalize(tuple(my_data.data_means), tuple(my_data.data_stds))
            ]
        )
        
        self.target_class = self.configuration['target_class']
        self.non_target_class = self.target_class if self.configuration['non_target_class'] is None else self.configuration['non_target_class']
        
        if self.model is None:
            self._model, self._last_layer = get_wrapped_model(
                self.original_model, self.target_class, return_last_layer=True, 
                normalization_wrap=self.configuration['normalization_wrap'],
                temperature_wrap=self.configuration['temperature_wrap']
            )
            self.model = copy.deepcopy(self.original_model)
            self.model.model = self._model.model
        
        if self.data_loader is None:
            self.print_out('Preparing new data loader.')
            self.active_data = self.get_data_with_targets(self.data, _type='train', redefine_transformation=self.configuration['normalization_wrap'])
            self.prepare_clean_and_poison_indices()
            self.available_data, self.data_loader = self.get_data_subset_personal(self.active_data, self.non_target_class, bs=self.batch_size)
            
        self.prepare_npca_things()
        
        if self.data_loader_original is None:
            self.data_loader_original = copy.deepcopy(self.data_loader)
        
        return
    
    
    def prepare_clean_and_poison_indices(self):
        # ##############################################
        # create lists of clean and poison indices
        # This should only be availble in the debug mode
        # ##############################################
        try:
            self._z_indices = np.where(np.array(self.active_data.targets)==self.non_target_class)[0]
            self.clean_indices, self.poison_indices = [], []
            for i, k in enumerate(self._z_indices):
                if k in self.active_data.poison_indices:
                    self.poison_indices.append(i)
                else:
                    self.clean_indices.append(i)
        except:
            self._z_indices = []
            self.clean_indices = []
            self.poison_indices = []
        return
    
    
    def pca_functions(self, values: np.ndarray, *args, **kwargs)-> General_PCA:
        return self.pca_classes[self.configuration['pca_type']](values, **kwargs)
    
    
    def get_data_with_targets(self, my_data: Torch_Dataset, _type='test', redefine_transformation: bool=True):
        
        self.print_out(my_data.data_name, 'kaggle_imagenet' not in my_data.data_name)
        
        _data = my_data.train if _type=='train' else my_data.test
        if redefine_transformation:
            my_data.update_transforms(self.redefined_test_transform, subdata_category=_type)
        
        return _data


    def __get_data_subset(self, imagenet, class_id, bs=128, num_workers=0):
        
        subset_idcs = torch.zeros(len(imagenet), dtype=torch.bool)
        in_targets = torch.LongTensor(imagenet.targets)
        subset_idcs[in_targets == class_id] = 1

        subset_idcs = torch.nonzero(subset_idcs, as_tuple=False).squeeze()
        in_subset = torch.utils.data.Subset(imagenet, subset_idcs)
        
        loader = torch.utils.data.DataLoader(in_subset, batch_size=bs, shuffle=False, num_workers=num_workers)
        
        return loader


    def get_data_subset_personal(self, imagenet, class_id, bs=128, num_workers=0, **kwargs):
        
        from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
        
        if class_id is None:
            _poison_in = np.where(torch.LongTensor(imagenet.targets) == self.target)[0]
        else:
            _poison_in = np.where(torch.LongTensor(imagenet.targets) == class_id)[0]
            
        in_subset = Client_SubDataset(imagenet, _poison_in)
        loader = torch.utils.data.DataLoader(in_subset, batch_size=bs, shuffle=False, num_workers=num_workers)
        
        return in_subset, loader
    
    
    def prepare_npca_things(self):
        
        self.ac_ = get_outputs(self._model, self.data_loader)
        self._pca = self.pca_functions(self.ac_.detach().cpu().numpy(), n_components=self.configuration['n_components'])
        
        return


    def alpha_metric(self, pca_dim, device: str=torch_device, metric_name: str='alpha_conf'):
        
        if metric_name == 'alpha':
            objective = self._pca.transform(self.ac_, device=self.model.device)[:, pca_dim]
        else:
            representation = self._pca.transform(self.ac_, device=self.model.device)
            logits = self.ac_.to(device)@self._last_layer.weight.T + self._last_layer.bias
            objective = representation[:, pca_dim] - torch.log(torch.sum(torch.exp(logits))) 
        
        return objective
        

    def maximizing_train_points(
        self, pca_dim, k=5, metric_name: str='alpha_conf',
        device: str=torch_device, return_indices: bool=False,
        use_original_dataloader: bool=True
    ):
        
        red_box = torch.zeros_like(self.data.train.__getitem__(0)[0])
        border_size = max(int(min(red_box.shape[1:])*2/32), 2)
        red_box = red_box.permute(1,2,0)
        red_box[-border_size:, :, :] = -1
        red_box[-border_size:, :, 0] = 1
        red_box = red_box.permute(1,0,2)
        red_box = red_box.permute(2, 1, 0)
        
        objective = self.alpha_metric(pca_dim, device=device, metric_name=metric_name)
        sorted_idcs = torch.argsort(objective, descending=True)
        
        # transform = self.data.default_train_transform
        self.data.update_transforms(self.redefined_test_transform, subdata_category='train')
        
        max_idcs = sorted_idcs[:k]
        max_images = []
        for idx in max_idcs:
            image = self.data_loader_original.dataset[idx][0] if use_original_dataloader else self.data_loader.dataset[idx][0]
            # if idx in self.poison_indices:
            #     image_ = image + red_box*(torch.max(image)-torch.min(image))
            #     image = torch.clamp(image_, torch.min(image), torch.max(image))
            max_images.append(image)
            
        self.data.reset_transforms()
            
        if return_indices:
            return max_images, max_idcs
        return max_images


    def analyze_model(self, *args, **kwargs):
        
        scores = np.zeros((len(self.data_loader.dataset)))
        feature_scores = np.zeros((len(self._pca.components)))
        good_features = [1]
        bad_features = [1]
                
        return scores, feature_scores, good_features, bad_features
    
    
    # Plot components
    def plot_components(
        self,
        num_features: int=5,
        save_fig: bool=False, results_dir: str='',
        num_images_for_each_feature: int=5,
        wrap_and_normalize_model: bool=True,
        show_features: list[int] = None,
        use_original_dataloader: bool=True
    ):
        """
        Inputs:
            show_features: Which features to show after the component analysis is done. If None, first num_features will be shown.
        """
        
        self.ac_ = torch.tensor(self.ac_).to(self.model.device) if isinstance(self.ac_, np.ndarray) else self.ac_.to(self.model.device)
        
        # here model
        x, y = get_data_samples_from_loader(self.data_loader, return_numpy=True)
        # x, y = x.numpy(), y.numpy()
        self.print_out(f'\n{evaluate_on_numpy_arrays(self.model.model, x, y, batch_size=self.batch_size):.3f}')
        torch.cuda.empty_cache()
        
        targets = np.arange(num_features) if show_features is None else show_features
        with torch.no_grad():
            adv_attack = Activation_Based_PGD(
                self._model, 
                eps=30, 
                n_iter=200, 
                norm='L2', 
                loss='obj_full', 
                verbose=False, 
                last_layer=self._last_layer,
                eigenvecs=self._pca.components,
                target_cls=self.target_class,
                device=self.model.device,
                # parameters below can also be commented and the algorithm will work just fine
                n_restarts=1, 
                seed=0,
                reg_other=1.,
                ica_components=None,
                ica_mean=None,
                minimize=False,
                minimize_abs=False
            )
            
            generated_features =  adv_attack.perturb(
                0.5 * torch.ones([len(targets)]+list(x[0].shape), device=self.model.device),
                torch.tensor(targets, dtype=torch.long), 
                best_loss=True,
                x_init=None
            )[0].detach()
            
            generated_features = generated_features if isinstance(generated_features, np.ndarray) else generated_features.detach().cpu().numpy()
            generated_features = generated_features if wrap_and_normalize_model else normalize(generated_features)


        # Compute maximal activating training images
        top_idcs = np.arange(num_features) if show_features is None else show_features
        n_max_imgs = num_images_for_each_feature

        max_cam_greyscales = {}
        max_imgs = {}
        for pc_idx in top_idcs:
            max_cam_greyscales[pc_idx] = []
            max_imgs[pc_idx] = self.maximizing_train_points(pc_idx, k=n_max_imgs, metric_name='alpha', return_indices=False, use_original_dataloader=use_original_dataloader)
        
        fontsize = 40
        fig_scaling = 3
        fig, ax = plt.subplots(len(top_idcs), n_max_imgs, figsize=(n_max_imgs*fig_scaling, 1*len(top_idcs)*fig_scaling), constrained_layout=True)
        for i, pc_idx in enumerate(top_idcs):
            # Feature Attack
            for j, max_img in enumerate(max_imgs[pc_idx]):
                # Max. activating training image
                ax[i][j].imshow(normalize(max_img.permute(1,2,0).numpy()) if max_img.shape[0]==3 else normalize(max_img[0].numpy()))
                if j!= 0: ax[i][j].axis('off')
            ax[i][0].set_ylabel(f'PC {pc_idx}', fontsize=fontsize)
            ax[i][0].tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
        plt.tight_layout()
        
        # fontsize = 20
        # fig_scaling = 3
        # fig, ax = plt.subplots(2, 2*len(top_idcs), figsize=(2*len(top_idcs)*fig_scaling, 2*fig_scaling), constrained_layout=True)
        # for i, pc_idx in enumerate(top_idcs):
        #     # Feature Attack
        #     max_img_s = max_imgs[pc_idx][:4]
        #     for j, max_img in enumerate(max_img_s):
        #         # Max. activating training image
        #         ax[(j//2)][2*i+j%2].imshow(normalize(max_img.permute(1,2,0).numpy()) if max_img.shape[0]==3 else normalize(max_img[0].numpy()))
        #         ax[(j//2)][2*i+j%2].axis('off')
        #     # ax[i][0].set_ylabel(f'PC {pc_idx}', fontsize=fontsize)
        #     # ax[0][0].tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
        # plt.tight_layout()
        
        fig_dir = f'p4_discovering_backdoors/__ignore__/figures'
        fig_name = f'{self.data.data_name}'
        fig_name += f'_(model={self.model.model_configuration['dataset_name']})'
        fig_name += f'_{self.defense_name}'
        if isinstance(self.data, Simple_Backdoor):
            fig_name += f'_(attack={self.data.attack_name})'
            fig_name += f'_(pr={self.data.poison_ratio})'
        fig_name += f'_(target={self.target_class})'
        if save_fig:
            plt.savefig(f'{fig_dir}/{fig_name}.pdf')
            
        return fig
    
    
    # Plot components
    def plot_components_with_features(
        self,
        num_features: int=5,
        save_fig: bool=False, results_dir: str='',
        num_images_for_each_feature: int=5,
        wrap_and_normalize_model: bool=True,
        show_features: list[int] = None,
        use_original_dataloader: bool=True
    ):
        """
        Inputs:
            show_features: Which features to show after the component analysis is done. If None, first num_features will be shown.
        """
        
        self.ac_ = torch.tensor(self.ac_).to(self.model.device) if isinstance(self.ac_, np.ndarray) else self.ac_.to(self.model.device)
        
        # here model
        x, y = get_data_samples_from_loader(self.data_loader, return_numpy=True)
        # x, y = x.numpy(), y.numpy()
        self.print_out(f'\n{evaluate_on_numpy_arrays(self.model.model, x, y, batch_size=self.batch_size):.3f}')
        torch.cuda.empty_cache()
        
        targets = np.arange(num_features) if show_features is None else show_features
        with torch.no_grad():
            adv_attack = Activation_Based_PGD(
                self._model, 
                eps=30, 
                n_iter=200, 
                norm='L2', 
                loss='obj_full', 
                verbose=False, 
                last_layer=self._last_layer,
                eigenvecs=self._pca.components,
                target_cls=self.target_class,
                device=self.model.device,
                # parameters below can also be commented and the algorithm will work just fine
                n_restarts=1, 
                seed=0,
                reg_other=1.,
                ica_components=None,
                ica_mean=None,
                minimize=False,
                minimize_abs=False
            )
            
            generated_features =  adv_attack.perturb(
                0.5 * torch.ones([len(targets)]+list(x[0].shape), device=self.model.device),
                torch.tensor(targets, dtype=torch.long), 
                best_loss=True,
                x_init=None
            )[0].detach()
            
            generated_features = generated_features if isinstance(generated_features, np.ndarray) else generated_features.detach().cpu().numpy()
            generated_features = generated_features if wrap_and_normalize_model else normalize(generated_features)


        # Compute maximal activating training images
        top_idcs = np.arange(num_features) if show_features is None else show_features
        n_max_imgs = num_images_for_each_feature

        max_cam_greyscales = {}
        max_imgs = {}
        for pc_idx in top_idcs:
            max_cam_greyscales[pc_idx] = []
            max_imgs[pc_idx] = self.maximizing_train_points(pc_idx, k=n_max_imgs, metric_name='alpha', return_indices=False, use_original_dataloader=use_original_dataloader)
        
        fontsize = 20
        fig_scaling = 3
        fig, ax = plt.subplots(len(top_idcs), 1 + n_max_imgs, figsize=(n_max_imgs*fig_scaling, 0.8*len(top_idcs)*fig_scaling), constrained_layout=True)
        for i, pc_idx in enumerate(top_idcs):
            # Feature Attack
            ax[i][0].imshow(np.transpose(generated_features[i], (1,2,0)))
            ax[i, 0].set_ylabel(f'PC {pc_idx}', fontsize=fontsize)
            ax[i][0].tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
            for j, max_img in enumerate(max_imgs[pc_idx]):
                # Max. activating training image
                ax[i][j+1].imshow(normalize(max_img.permute(1,2,0).numpy()) if max_img.shape[0]==3 else normalize(max_img[0].numpy()))
                ax[i][j+1].axis('off')
        plt.tight_layout()
        
        fig_name = f'{self.data.data_name}'
        if isinstance(self.data, Simple_Backdoor):
            fig_name += f'_(attack={self.data.attack_name})'
            fig_name += f'_(pr={self.data.poison_ratio})'
        plt.savefig(f'{fig_name}.pdf')
            
        return fig
    
    
    def remove_feature(
        self, 
        features_to_remove: int=None, features_to_keep: int=None,
        epochs: int=1, learning_rate: float=1e-1,
        use_original_dataloader: bool=True, device: str=torch_device
    ):
        
        if (features_to_remove is None) and (features_to_keep is None):
            self.print_out('You have not input any features to either enforce or remove.')
            return
        
        def project_by_removing_feature(_mnpca: NPCA_Paper, activations: np.ndarray, features_to_remove: list[int]=None, features_to_keep: list[int]=None):
            component_values = copy.deepcopy(_mnpca._pca.transform(activations))
            all_features = np.zeros((1, component_values.shape[1]))
            if features_to_remove is not None:
                for remove_feature in features_to_remove:
                    # all_features[0, remove_feature] = -2 * max(np.abs(np.min(component_values)), np.abs(np.max(component_values)))
                    component_values[:, remove_feature] = np.clip(component_values[:, remove_feature], np.min(component_values), -1)
            if features_to_keep is not None:
                for keep_feature in features_to_keep:
                    # all_features[0, keep_feature] = 2 * max(np.abs(np.min(component_values)), np.abs(np.max(component_values)))
                    component_values[:, keep_feature] = np.clip(component_values[:, keep_feature], 1, np.max(component_values))
            return _mnpca._pca.inverse_transform(component_values)
        
        
        # prepare logits for the model
        removed_activations = project_by_removing_feature(self, self.ac_.detach().cpu().numpy(), features_to_remove, features_to_keep=features_to_keep)
        x, _ = self.sample_from_dataloader(return_numpy=True, use_original_dataloader=use_original_dataloader)
        spufix_dataloader = prepare_dataloader_from_numpy(x, removed_activations, batch_size=self.batch_size, shuffle=True)
        
        
        # #############################
        # Removing feature now
        # #############################
        # optimizer = torch.optim.Adam(self._model.last_layer.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            
            loss_over_data = 0
            self.model.model.train()
            self._model.train()
            for batch_idx, (sample, target) in enumerate(spufix_dataloader):
                torch.cuda.empty_cache()
                sample, target = sample.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                output = self._model(sample)
                output_loss = torch.abs(output - target)**2
                loss = torch.mean(output_loss)
                
                loss.backward()
                optimizer.step()
                
                loss_over_data += loss.item()
                
                print_str = f'Epoch: {epoch}({100. * batch_idx / len(spufix_dataloader):3.1f}%] | tr_loss: {loss_over_data / min( (batch_idx+1) * spufix_dataloader.batch_size, len(spufix_dataloader.dataset) ):.5f}'
                self.print_out('\r' + print_str, end='', verbose=True)
        self._model.eval()
        self.model.model.eval()
                
        return
    
    
    def sample_from_dataloader(self, return_numpy: bool=True, use_original_dataloader: bool=False, **kwargs):
        if use_original_dataloader:
            inputs, y = get_data_samples_from_loader(self.data_loader_original, return_numpy=return_numpy, **kwargs)
        else:
            inputs, y = get_data_samples_from_loader(self.data_loader, return_numpy=return_numpy, **kwargs)
        return inputs, y
    
    
    def plot_a_metric(self, metric, clean_indices, poison_indices):
        
        plt.figure()
        plt.hist(metric, label='All', bins=100, alpha=0.5)
        plt.hist(metric[clean_indices], label='Clean', color='green', bins=100, alpha=0.5)
        plt.hist(metric[poison_indices], label='Backdoor', color='red', bins=100, alpha=0.5);
        plt.show()
        plt.close()
        
        return
    
    
    def compute_accuracy_of_differentiation(self, metric, clean_indices, poison_indices, num_clusters: int=2):
        
        def pairwise_cosine_similarity_torch(flattened_clients_states):
            normalized_input_a = torch.nn.functional.normalize(flattened_clients_states)
            res = torch.mm(normalized_input_a, normalized_input_a.T)
            res[res==0] = 1e-6
            return res
        
        def linearly_normalized_torch(arr_in: torch.Tensor):
            return (arr_in-torch.min(arr_in))/(torch.max(arr_in)-torch.min(arr_in)) if torch.max(arr_in)>torch.min(arr_in) else arr_in/torch.max(arr_in)
        
        def _get_f1_score_from_labels(one_indices, labels):
            y_true = np.zeros((len(labels)))
            y_true[one_indices] = 1
            return f1_score(y_true, labels), accuracy_score(y_true, labels)
        
        def get_f1_score_from_labels(one_indices, labels):
            
            unique_labels = np.unique(labels)
            
            f1_scores, accs = [], []
            for i in unique_labels:
                new_labels = np.zeros((len(labels)))
                new_labels[np.where(labels==i)] = 1
                f1_i, acc_i = _get_f1_score_from_labels(one_indices, new_labels)
                
                f1_scores.append(f1_i), accs.append(acc_i)
                
            return np.array(f1_scores), np.array(accs)
        
        
        self_cs_values = pairwise_cosine_similarity_torch( torch.tensor(metric) )
        self_cs_values = linearly_normalized_torch(self_cs_values)
        
        # convert Pytorch values into numpy for later use
        self.metric_values = self_cs_values.detach().cpu().numpy()
        
        spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
        kmeans = KMeans(n_clusters=num_clusters)
        self.clusterer = kmeans
        self.clusterer.fit(self.metric_values)
        self.labels_ = self.clusterer.labels_
        
        f1_scores, accs = get_f1_score_from_labels(clean_indices, self.labels_)
        # f1_b, acc_b = get_f1_score_from_labels(poison_indices, self.labels_)
        self.print_out('Metrics are:\n' + '\n'.join([f'F1_{i}={f1_scores[i]:.3f}, acc_{i}={accs[i]:.3f}' for i in range(len(f1_scores))]))
        
        return f1_scores, accs
    
    
    # Plot components
    def plot_components_averaged(
        self,
        show_features: list[int] = [0,1,2],
        num_images_for_each_feature: int=5,
        wrap_and_normalize_model: bool=True,
        use_original_dataloader: bool=True
    ):
        """
        Inputs:
            show_features: Which features to show after the component analysis is done. If None, first num_features will be shown.
        """
        
        self.ac_ = torch.tensor(self.ac_).to(self.model.device) if isinstance(self.ac_, np.ndarray) else self.ac_.to(self.model.device)
        
        # here model
        x, y = get_data_samples_from_loader(self.data_loader, return_numpy=True)
        # x, y = x.numpy(), y.numpy()
        self.print_out(f'\n{evaluate_on_numpy_arrays(self.model.model, x, y, batch_size=self.batch_size):.3f}')
        torch.cuda.empty_cache()
        
        targets = np.arange(num_features) if show_features is None else show_features
        with torch.no_grad():
            adv_attack = Activation_Based_PGD(
                self._model, 
                eps=30, 
                n_iter=200, 
                norm='L2', 
                loss='obj_full', 
                verbose=False, 
                last_layer=self._last_layer,
                eigenvecs=self._pca.components,
                target_cls=self.target_class,
                device=self.model.device,
                # parameters below can also be commented and the algorithm will work just fine
                n_restarts=1, 
                seed=0,
                reg_other=1.,
                ica_components=None,
                ica_mean=None,
                minimize=False,
                minimize_abs=False
            )
            
            generated_features =  adv_attack.perturb(
                0.5 * torch.ones([len(targets)]+list(x[0].shape), device=self.model.device),
                torch.tensor(targets, dtype=torch.long), 
                best_loss=True,
                x_init=None
            )[0].detach()
            
            generated_features = generated_features if isinstance(generated_features, np.ndarray) else generated_features.detach().cpu().numpy()
            generated_features = generated_features if wrap_and_normalize_model else normalize(generated_features)


        # Compute maximal activating training images
        top_idcs = show_features
        n_max_imgs = num_images_for_each_feature

        # max_cam_greyscales = {}
        # max_imgs = {}
        # for pc_idx in top_idcs:
        max_imgs = self.maximizing_train_points_averaged(show_features, k=n_max_imgs, metric_name='alpha_conf', return_indices=False, use_original_dataloader=use_original_dataloader)
        
        fontsize = 20
        fig_scaling = 3
        fig, ax = plt.subplots(1, n_max_imgs, figsize=(n_max_imgs*fig_scaling, 0.8*fig_scaling), constrained_layout=True)
        # for i, pc_idx in enumerate(top_idcs):
        # Feature Attack
        # ax[0].imshow(np.transpose(generated_features, (1,2,0)))
        ax[0].tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
        for j, max_img in enumerate(max_imgs):
            # Max. activating training image
            ax[j].imshow(normalize(max_img.permute(1,2,0).numpy()) if max_img.shape[0]==3 else normalize(max_img[0].numpy()))
            ax[j].axis('off')
        plt.title(f'PC {show_features}', fontsize=fontsize)
        plt.tight_layout()
            
        return fig
    
    
    def maximizing_train_points_averaged(
        self, pca_dim_s, k=5, metric_name: str='alpha_conf',
        device: str=torch_device, return_indices: bool=False,
        use_original_dataloader: bool=True
    ):
        
        objective = torch.stack([self.alpha_metric(pca_dim, device=device, metric_name=metric_name) for pca_dim in pca_dim_s], dim=0)
        objective = torch.mean(objective, dim=0)
        sorted_idcs = torch.argsort(objective, descending=True)
        
        max_idcs = sorted_idcs[:k]
        max_images = []
        for idx in max_idcs:
            max_images.append(self.data_loader_original.dataset[idx][0] if use_original_dataloader else self.data_loader.dataset[idx][0])
        if return_indices:
            return max_images, max_idcs
        
        return max_images