import numpy as np

import torch
import torchvision

import matplotlib.pyplot as plt


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from ..test.adversarial_input_manipulations.activation_based_pgd import Activation_Based_PGD
from ..model_utils.feature_activations import Feature_Activations

# from .._imported_implementations_.spurious_imagenet.neural_pca.adversarial_attacks.act_apgd import ActivationAPGDAttack
# from .._imported_implementations_.spurious_imagenet.utils.temperature_wrapper import TemperatureWrapper
# from .._imported_implementations_.spurious_imagenet.utils.model_normalization import ImageNetWrapper

from ..model_utils.loss_functions.pca import PCA_Loss, PCA_of_SKLEARN, PCA_of_NPCA
from ..model_utils.wrapping_utils import get_wrapped_model

from utils_.torch_utils import get_outputs, get_data_samples_from_loader, evaluate_on_numpy_arrays
from utils_.general_utils import normalize



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_data_with_targets(my_data: Torch_Dataset, _type='test', redefine_transformation: bool=True):
    
    if my_data.data_name != 'kaggle_imagenet':
        _data = my_data.train if _type=='train' else my_data.test
        _data.targets = [_data[i][1] for i in range(_data.__len__())]
    else:
        _data = my_data.full_dataset
    
    if redefine_transformation:
        test_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(my_data.preferred_size), torchvision.transforms.ToTensor()]
        )
        _data.transform = test_transform
        
    return _data


def get_data_subset(imagenet, class_id, bs=128, num_workers=1):
    
    subset_idcs = torch.zeros(len(imagenet), dtype=torch.bool)
    in_targets = torch.LongTensor(imagenet.targets)
    subset_idcs[in_targets == class_id] = 1

    subset_idcs = torch.nonzero(subset_idcs, as_tuple=False).squeeze()
    in_subset = torch.utils.data.Subset(imagenet, subset_idcs)
    
    loader = torch.utils.data.DataLoader(in_subset, batch_size=bs, shuffle=False, num_workers=num_workers)
    
    return loader


def get_data_subset_personal(imagenet, class_id, bs=128, num_workers=1):
    
    from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
    
    _poison_in = np.where(torch.LongTensor(imagenet.targets) == class_id)[0]
    in_subset = Client_SubDataset(imagenet, _poison_in)
    
    loader = torch.utils.data.DataLoader(in_subset, batch_size=bs, shuffle=False, num_workers=num_workers)
    
    return loader


def perform_pca_transformation(activations, mean, components, device: str=torch_device, eigvec_scale: bool=True):
        
        # act = activations - self.pca_mean
        act = activations.to(device) - torch.tensor(mean).to(device)
        act_pca = act@torch.tensor(components, dtype=torch.float32).to(device)
        if eigvec_scale:
            act_pca = act_pca * torch.sum(torch.tensor(components, dtype=torch.float32).to(device), dim=0)
        
        return act_pca
    

def maximizing_train_points_alpha_conf(
    data_loader: torch.utils.data.DataLoader, 
    ac_, _last_layer, pca_dim, mean, components,
    k=5,
    device: str=torch_device, return_indices: bool=False
):
    
    representation = perform_pca_transformation(ac_, mean, components, device=device)
    logits = ac_.to(device)@_last_layer.weight.T + _last_layer.bias
    objective = representation[:, pca_dim] - torch.log(torch.sum(torch.exp(logits))) 
    sorted_idcs = torch.argsort(objective, descending=True)
    
    max_idcs = sorted_idcs[:k]
    max_images = []
    for idx in max_idcs:
        max_images.append(data_loader.dataset[idx][0])
    if return_indices:
        return max_images, max_idcs
    
    return max_images


def maximizing_train_points_alpha(
    data_loader: torch.utils.data.DataLoader,
    ac_, _last_layer, pca_dim, mean, components,
    k=5,
    device: str=torch_device, return_indices=False
):
    
    representation = perform_pca_transformation(ac_, mean, components, device=device)
    # objective = representation[:, pca_dim]
    sorted_idcs = torch.argsort(representation[:, pca_dim], descending=True)
    max_idcs = sorted_idcs[:k]
    max_images = []
    for idx in max_idcs:
        max_images.append(data_loader.dataset[idx][0])
    if return_indices:
        return max_images, max_idcs
    
    return max_images


# Plot components
def plot_components(
    torch_model: Torch_Model,
    target_class: int=94, non_target_class: int=None,
    num_features: int=5,
    save_fig: bool=False, results_dir: str='',
    data_loader = None,
    wrap_and_normalize_model: bool=True,
    show_features: list[int] = None
):
    """
    Inputs:
        show_features: Which features to show after the component analysis is done. If None, first num_features will be shown.
    """
    
    non_target_class = target_class if non_target_class is None else non_target_class
    
    _model, _last_layer = get_wrapped_model(torch_model, target_class, return_last_layer=True)
    
    if data_loader is None:
        data_to_consider = get_data_with_targets(torch_model.data, _type='train', redefine_transformation=wrap_and_normalize_model)
        # data_loader = get_data_subset(data_to_consider, non_target_class, bs=torch_model.model_configuration['batch_size'])
        data_loader = get_data_subset_personal(data_to_consider, non_target_class, bs=torch_model.model_configuration['batch_size'])
    
    x, y = get_data_samples_from_loader(data_loader)
    ac_ = get_outputs(_model, data_loader)
    _pca = PCA_of_NPCA(ac_.detach().cpu().numpy().T)
    components, mean = _pca.components, _pca.mean
    x, y = x.numpy(), y.numpy()
    print(f'\n{evaluate_on_numpy_arrays(torch_model.model, x, y, batch_size=torch_model.model_configuration['batch_size']):.3f}')
    torch.cuda.empty_cache()
    
    targets = np.arange(num_features) if show_features is None else show_features
    with torch.no_grad():
        adv_attack = Activation_Based_PGD(
            _model, 
            eps=30, 
            n_iter=200, 
            norm='L2', 
            loss='obj_full', 
            verbose=False, 
            last_layer=_last_layer,
            eigenvecs=components,
            target_cls=target_class,
            device=torch_model.device,
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
            0.5 * torch.ones([len(targets)]+list(x[0].shape), device=torch_model.device),
            torch.tensor(targets, dtype=torch.long), 
            best_loss=True,
            x_init=None
        )[0].detach()
        
        generated_features = generated_features if isinstance(generated_features, np.ndarray) else generated_features.detach().cpu().numpy()
        generated_features = generated_features if wrap_and_normalize_model else normalize(generated_features)


    # Compute maximal activating training images
    top_idcs = np.arange(num_features) if show_features is None else show_features
    n_max_imgs = 5

    max_cam_greyscales = {}
    max_imgs = {}
    for pc_idx in top_idcs:
        max_cam_greyscales[pc_idx] = []
        max_imgs[pc_idx] = maximizing_train_points_alpha_conf(data_loader, ac_, _last_layer, pc_idx, mean, components, k=n_max_imgs, return_indices=False)
    
    
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



# Plot components
def later_blabla(
    torch_model: Torch_Model, _last_layer, 
    data_loader, sample,
    ac_, mean, components,
    target_class: int=94, non_target_class: int=None,
    num_features: int=5,
    save_fig: bool=False, results_dir: str=''
):
    
    non_target_class = target_class if non_target_class is None else non_target_class
    
    targets = np.arange(num_features)
    with torch.no_grad():
        adv_attack = Activation_Based_PGD(
            torch_model.model, 
            eps=30, 
            n_iter=200, 
            norm='L2', 
            loss='obj_full', 
            verbose=False, 
            last_layer=_last_layer,
            eigenvecs=components,
            target_cls=target_class,
            device=torch_model.device,
            # parameters below can also be commented and the algorithm will work just fine
            n_restarts=1, 
            seed=0,
            reg_other=1.,
            ica_components=None,
            ica_mean=None,
            minimize=False,
            minimize_abs=False
        )
        
        _cfs =  adv_attack.perturb(
            0.5 * torch.ones([len(targets)]+list(sample.shape), device=torch_model.device),
            torch.tensor(np.arange(5), dtype=torch.long), 
            best_loss=True,
            x_init=None
        )[0].detach()


    # Compute maximal activating training images
    top_idcs = np.arange(5)
    n_max_imgs = 5

    cf_cam_greyscales = []
    max_cam_greyscales = {}
    max_imgs = {}
    max_idcs = {}
    for pc_idx in top_idcs:
        max_cam_greyscales[pc_idx] = []
        max_imgs[pc_idx] = maximizing_train_points_alpha_conf(data_loader, ac_, _last_layer, pc_idx, mean, components, k=n_max_imgs, return_indices=False)
    
    
    fontsize = 20
    fig_scaling = 3
    for i, pc_idx in enumerate(top_idcs):
        fig, ax = plt.subplots(1, 1 + n_max_imgs, figsize=(n_max_imgs*fig_scaling, fig_scaling), constrained_layout=True)
        fig.suptitle(f'Class {target_class} - Component {i+1} ({pc_idx} by eigenval)', fontsize=fontsize)

        # Feature Attack
        ax[0].imshow(_cfs[pc_idx].detach().cpu().permute(1,2,0))
        ax[0].set_ylabel(f'PC {pc_idx}', fontsize=fontsize)
        ax[0].tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
        
        for j, max_img in enumerate(max_imgs[pc_idx]):
            # Max. activating training image
            ax[j+1].imshow(max_img.permute(1,2,0))
            ax[j+1].axis('off')
        
        if save_fig:
            print('Saving fig...')
            plt.savefig(f'{results_dir}/{i+1}_visualize_component_{pc_idx}.png')
        plt.show()
        
    return


