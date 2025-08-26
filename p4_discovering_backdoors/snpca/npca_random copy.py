import numpy as np

import torch
import torchvision
import copy

import matplotlib.pyplot as plt

from termcolor import colored

from sklearn.cluster import SpectralClustering, KMeans, HDBSCAN
from sklearn.metrics import f1_score, accuracy_score


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model
from _0_general_ML.data_utils.datasets import Fashion_MNIST, GTSRB, CIFAR10

# from .npca_paper import NPCA_Paper
from .npca_custom_with_masking import NPCA_Custom_with_Masking
from .one_channel_data import Channel1_Torch_Dataset

from utils_.torch_utils import get_outputs, get_data_samples_from_loader, evaluate_on_numpy_arrays, prepare_dataloader_from_numpy
from utils_.general_utils import normalize, exponential_normalize

from ..attacks.input_minimalist import Input_Minimalist
from ..attacks.input_minimalist_patch import Patch_Input_Minimalist
from ..attacks.fgsm_attack import FGSM_with_Dict
from ..attacks.adv_attack import Random_Patch_Adversarial_Attack
from ..attacks.activation_based_pgd import Activation_Based_PGD
from ..attacks.adv_attack_copy import Random_Patch_Invisible_Visible_Adversarial_Attack

from .analyzer_random import PCA_Analyzer_Random



torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def input_minimalist_functions(
    minimalist_type: str, model: Torch_Model, minimalist_configuration: dict
) -> Input_Minimalist:
    
    all_input_minimalists = {
        'pixel_based_masking': Input_Minimalist,
        'patch_based_masking': Patch_Input_Minimalist,
        'patch': Random_Patch_Adversarial_Attack,
        'fgsm': FGSM_with_Dict
    }
    
    return all_input_minimalists[minimalist_type](model, minimalist_configuration)


class NPCA_Random(NPCA_Custom_with_Masking):
    
    def __init__(
        self, 
        data: Torch_Dataset=None,
        torch_model: Torch_Model=None, 
        defense_configuration: dict={},
        use_new_data: bool=True,
        verbose: bool=True,
        **kwargs
    ):
        
        self.defense_name = 'SNPCA_Random'
        
        new_data = data
        if use_new_data:
            if 'gtsrb' not in torch_model.data.data_name:
                new_data = GTSRB(preferred_size=data.preferred_size, data_means=data.data_means, data_stds=data.data_stds)
            else:
                new_data = CIFAR10(preferred_size=data.preferred_size, data_means=data.data_means, data_stds=data.data_stds)
            if data.get_input_shape()[0] == 1:
                new_data = Channel1_Torch_Dataset(new_data)
        self.num_classes = torch_model.data.num_classes
        
        default_defense_configuration = {
            'fgsm_configuration': {
                'loss': 'crossentropy',
                'epsilon': 0.2,
                'iterations': 50,
            },
            'patch_configuration': {
                'loss': 'crossentropy',
                'mask_ratio': 0.4,
                # 'number_of_masks': 3,
                'iterations': 100,
            },
        }
        for key in default_defense_configuration.keys():
            if key not in defense_configuration.keys():
                defense_configuration[key] = default_defense_configuration[key]
        
        super().__init__(data=new_data, torch_model=torch_model, defense_configuration=defense_configuration, verbose=verbose)
        
        return
    
    
    def prepare_analyzer(self, *args, mode: str='one', threshold: float=0, **kwargs):
        # self.torch_model.data.update_transforms(self.redefined_test_transform)
        _xq, y = get_data_samples_from_loader(torch.utils.data.DataLoader(self.torch_model.data.test, batch_size=self.data_loader.batch_size), return_numpy=True)
        transformed_loader = prepare_dataloader_from_numpy(_xq, y, batch_size=self.data_loader.batch_size)
        # self.torch_model.data.reset_transforms()
        return PCA_Analyzer_Random(*args, mode='one', threshold=0, reference_loader=transformed_loader, **kwargs)
    
    
    def prepare_masked_data(self):
        
        self.prepare_redefined_transformed_loader()
        
        _x, y_random = get_data_samples_from_loader(self.data_loader, return_numpy=True)
        perturbed_x = _x.copy()
        
        # generating targeted labels
        y_targeted = self.target_class*np.ones_like(y_random)
        
        # generating random targets but not the targets of the target class
        y_random = np.random.randint(0, self.num_classes, size=len(y_targeted))
        random_additions = np.random.randint(1, self.num_classes-1, size=len(y_targeted))
        y_random[y_random==self.target_class] += random_additions[y_random==self.target_class]
        y_random = y_random % self.num_classes
        print(colored(f'The data name is {self.data.data_name}, model data is {self.model.data.data_name} and classes num is {self.num_classes}', 'red'))
        print(colored(f'The max random class is {np.max(y_random)}, classes num is {self.num_classes}', 'red'))
        # assert False
        
        # perturbing random images
        fgsm_attack = input_minimalist_functions('fgsm', self.model, self.configuration['fgsm_configuration'])
        fgsm_perturbations = fgsm_attack.attack(perturbed_x, y_random, iterations=self.configuration['fgsm_configuration']['iterations'], targeted=True)
        perturbed_x = fgsm_attack.perturb(perturbed_x, fgsm_perturbations)
        self.data_loader_with_perturbation = prepare_dataloader_from_numpy(perturbed_x, y_random, batch_size=self.batch_size)
        
        # adding target class targeted patches to random locations of the perturbed images
        patch_attack = input_minimalist_functions('patch', self.model, self.configuration['patch_configuration'])
        patch_perturbations = patch_attack.attack(perturbed_x, y_targeted, iterations=self.configuration['patch_configuration']['iterations'])
        perturbed_x = patch_attack.perturb(perturbed_x, patch_perturbations).astype(np.float32)
        
        pred_y = get_outputs(self.model.model, prepare_dataloader_from_numpy(perturbed_x, y_targeted, batch_size=self.batch_size))
        pred_y = np.argmax(pred_y, axis=1)
        successful_indices = (pred_y == self.target_class)
        self.patches = patch_perturbations[successful_indices]
        perturbed_x, y_targeted = perturbed_x[successful_indices], y_targeted[successful_indices]
        
        print(len(pred_y), len(perturbed_x))
        # failed_x = perturbed_x[failed_indices]
        # new_fgsm_attack = FGSM_with_Dict(self.model, inversion_configuration={'epsilon': 0.1})
        # targeted_perturbations = new_fgsm_attack.attack(failed_x, y_targeted, iterations=self.configuration['fgsm_configuration']['iterations'])
        # perturbed_failed_x = new_fgsm_attack.perturb(failed_x, targeted_perturbations)
        # perturbed_x[failed_indices] = perturbed_failed_x
        
        # making dataloader original
        self.data_loader_original = prepare_dataloader_from_numpy(perturbed_x, y_targeted, batch_size=self.batch_size)
        
        # mask_attack = input_minimalist_functions('pixel_based_masking', self.model, self.configuration['masking_configuration'])
        mask_attack = input_minimalist_functions('patch_based_masking', self.model, self.configuration['masking_configuration'])
        masks = mask_attack.attack(perturbed_x, y_targeted, iterations=self.configuration['masking_configuration']['iterations'], targeted=True)
        perturbed_x = mask_attack.perturb(perturbed_x, masks).astype(np.float32)
        masked_data_loader = prepare_dataloader_from_numpy(perturbed_x.astype(np.float32), y_targeted, batch_size=self.batch_size)
        self.configure_defense(data_loader=masked_data_loader)
        self.prepare_npca_things_original()
        
        return
    
    
    def defend_mr(self, *args, epochs=10, **kwargs):
        
        self.model.train(epochs=epochs)
        self.defend()
        
        return
    
    
    def __defend(self, *args, adversarial_subset_ratio: float=0.1, use_original_dataloader: bool=True, show_plots=False, show_metrics=False, **kwargs):
        
        self.print_out(self.configuration, verbose=True)
        # state_before = self.key_flatten_client_state_np(self.model.model.state_dict())
        
        # self.analyze_model(use_original_dataloader=True, show_plots=show_plots, show_metrics=show_metrics)
        x, y = self.sample_from_dataloader(return_numpy=True, use_original_dataloader=True)
        # self.analyzer = PCA_Analyzer(self.model, self._model, x, y, self.ac_.detach().cpu().numpy(), mode='multi', threshold=0.)
        self.analyze_model(use_original_dataloader=True, show_plots=show_plots, show_metrics=show_metrics)
        
        self.final_model = self.model 
        
        return
        
    
    def adversarial_forgetting(self, adversarial_subset_ratio: float=0.1, learning_rate: float=1e-2, n_epochs: int=1, use_original_dataloader: bool=True, **kwargs):
        
        if self.adversarial_dataloader is None:
            data_len = int(len(self.data_loader.dataset)*adversarial_subset_ratio)
            
            indices_good = np.argsort(self.scores)[:data_len]
            indices_bad = np.argsort(-self.scores)[:data_len]
            
            x, y = self.sample_from_dataloader(return_numpy=True, use_original_dataloader=use_original_dataloader)
            xp, yp = get_data_samples_from_loader(self.data_loader_with_perturbation, return_numpy=True)
            self.print_out(np.min(x), np.max(x), x.shape, xp.shape)
            
            x_b = x[indices_bad]
            xp = xp[indices_bad]
            outputs = self.model.model(torch.tensor(xp[:data_len]).to(self.model.device)).detach().cpu().numpy()
            yp = np.argmax(outputs, axis=1)
            # print(yp)
            x_a, y_a = np.append(x_b, xp, axis=0), np.append(yp, yp, axis=0)
            
            self.adversarial_dataloader = prepare_dataloader_from_numpy(x_a, y_a, batch_size=self.batch_size, shuffle=True)
            self.bad_adversarial_dataloader = prepare_dataloader_from_numpy(x_b, self.target_class*np.ones_like(y_a[:len(x_b)]), batch_size=self.batch_size, shuffle=True)
        
        self.model.optimizer_class.lr = learning_rate
        self.model.optimizer = self.model.optimizer_class.return_optimizer(self.model.model.parameters())
        # self.model.optimizer = self.model.optimizer_class.return_optimizer(self._last_layer.parameters())
        
        for epoch in range(n_epochs):
            
            loss_over_data, acc_over_data = 0, 0
            self.model.model.train()
            for batch_idx, (sample, target) in enumerate(self.adversarial_dataloader):
                torch.cuda.empty_cache()
                sample, target = sample.to(self.model.device), target.to(self.model.device)
                
                self.model.optimizer.zero_grad()
                output = self.model.model(sample)
                loss = torch.mean(torch.nn.functional.cross_entropy(output, target))
                loss.backward()
                self.model.optimizer.step()
                
                loss_over_data += loss.item()
                pred = output.argmax(1, keepdim=True)
                acc_over_data += pred.eq(target.view_as(pred)).sum().item()
                
                print_str = f'Epoch: {epoch}({100. * batch_idx / len(self.adversarial_dataloader):3.1f}%] | tr_loss: {loss_over_data / min( (batch_idx+1) * self.adversarial_dataloader.batch_size, len(self.adversarial_dataloader.dataset) ):.5f}'
                self.print_out(f'\rForgetting: {print_str}', end='', verbose=True)
        
        self.model.model.eval()
        
        return
        
    
    def plot_components_with_target_class_data_loader(
        self, 
        num_features = 5, save_fig = False, results_dir = '', num_images_for_each_feature = 5, 
        wrap_and_normalize_model = True, show_features = None
    ):
        """
        Inputs:
            show_features: Which features to show after the component analysis is done. If None, first num_features will be shown.
        """
        
        def alpha_metric(pca_dim, ac_, device: str=torch_device, metric_name: str='alpha_conf'):
        
            if metric_name == 'alpha':
                objective = self._pca.transform(ac_, device=self.model.device)[:, pca_dim]
            else:
                representation = self._pca.transform(ac_, device=self.model.device)
                logits = ac_.to(device)@self._last_layer.weight.T + self._last_layer.bias
                objective = representation[:, pca_dim] - torch.log(torch.sum(torch.exp(logits))) 
            
            return objective
        
        def maximizing_train_points(
            pca_dim, data_loader=None, ac_=None,
            k=5, metric_name: str='alpha_conf',
            device: str=torch_device, return_indices: bool=False
        ):
            
            if (data_loader is None) or (ac_ is None):
                ac_ = self.ac_
                data_loader = self.data_loader
            
            objective = alpha_metric(pca_dim, ac_, device=device, metric_name=metric_name)
            sorted_idcs = torch.argsort(objective, descending=True)
            
            max_idcs = sorted_idcs[:k]
            max_images = []
            for idx in max_idcs:
                max_images.append(data_loader.dataset[idx][0])
            if return_indices:
                return max_images, max_idcs
            
            return max_images
        
        self.ac_ = torch.tensor(self.ac_).to(self.model.device) if isinstance(self.ac_, np.ndarray) else self.ac_.to(self.model.device)
        
        # here model
        _, data_loader_to_use = self.get_data_subset_personal(
            self.active_data, self.target_class, bs=self.batch_size,
            # redefine_transformation=self.configuration['normalization_wrap']
        )
        x, y = get_data_samples_from_loader(data_loader_to_use, return_numpy=True)
        ac_ = get_outputs(self._model, data_loader_to_use)
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
            max_imgs[pc_idx] = maximizing_train_points(pc_idx, data_loader=data_loader_to_use, ac_=ac_, k=n_max_imgs, return_indices=False, metric_name='alpha_conf')
        
        
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
                ax[i][j+1].imshow(normalize(max_img.permute(1,2,0).numpy()))
                ax[i][j+1].axis('off')
        plt.tight_layout()
            
        return fig
    
    
    