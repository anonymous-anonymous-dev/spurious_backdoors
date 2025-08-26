import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from .loading import load_results_from_settings
from utils_.general_utils import confirm_directory



nicer_names_for_figures = {
    'mnist': 'MNIST',
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
    
    'vanilla': 'No Defense',
    'strip': 'STRIP',
    'activation_clustering': 'AC',
    'spectral_signatures': 'SS',
    'mdtd': 'MDTD', 
    'zero_shot_purification': 'ZIP',
    'snpca_id': 'ASNPCA-I',
    'snpca_ood': 'ASNPCA-II',
    'snpca_ood_efficient': 'ASNPCA-II',
    
    'simple_backdoor_0.1': 'VTBA',
    'invisible_backdoor_0.1': 'ITBA',
    'reflection_backdoor_0.1': 'RBA',
    'clean_label_backdoor_0.1': 'LCBA',
    'wanet_backdoor_0.1': 'WBA',
    'horizontal_backdoor_0.1': 'HBA',
    
    'simple_backdoor_0.3': 'VTBA',
    'invisible_backdoor_0.3': 'ITBA',
    'reflection_backdoor_0.3': 'RBA',
    'clean_label_backdoor_0.3': 'LCBA',
    'wanet_backdoor_0.3': 'WBA',
    'horizontal_backdoor_0.3': 'HBA',
    
}



def save_figure_multiple_pages(figs: list, save_fig_path_and_name: str):
    confirm_directory( '/'.join(save_fig_path_and_name.split('/')[:-1]) )
    with PdfPages(save_fig_path_and_name) as p:
            for fig in figs:
                fig.savefig(p, format='pdf')
            print('figure saved.')
    return


def hyperparameter_pr_figure_comparison(dataset_names, results_path_local: str, backdoor_prefix: str='simple', suffix: str='DC', figure_name: str='hyperparameter_pr_comparison', save_fig: bool=False):
    
    suffix = '' if suffix=='DC' else '_'+suffix
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    poison_ratios = [0.01, 0.03, 0.05, 0.1, 0.3, 0.4, 0.5]
    backdoor_types = [f'{backdoor_prefix}_backdoor_{pr}' for pr in poison_ratios]
    
    defense_type_theirs = [
        'vanilla',
        'strip', 
        'activation_clustering', 
        'spectral_signatures', 
        'mdtd', 
        # 'zero_shot_purification',
    ]
    defense_type_ours = [
        'snpca_id' if suffix=='' else 'snpca_ood'
    ]
    defense_types = defense_type_theirs + defense_type_ours
    
    keys = ['ca', 'pa']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types, 
        keys=[f'{key}{suffix}' for key in keys],
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.round(results_arr, decimals=2)
    
    figs = []
    
    fig = plt.figure(figsize=(5, 2.5))
    for s, defense_type in enumerate(defense_types):
        plt.plot(results_arr[0, :, s, 0], marker='o', markerfacecolor='none', label=f'{nicer_names_for_figures[defense_type]}')
    plt.xlim([None, 10])
    plt.xticks(np.arange(len(poison_ratios)), np.array(poison_ratios)*10)
    plt.xlabel('Poison Ratio: PR (%)')
    plt.ylabel('Clean Accuracy: CA')
    plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    fig = plt.figure(figsize=(5, 2.5))
    for s, defense_type in enumerate(defense_types):
        plt.plot(results_arr[0, :, s, 1], marker='x', markerfacecolor='none', label=f'{nicer_names_for_figures[defense_type]}')
    plt.xlim([None, 10])
    plt.xticks(np.arange(len(poison_ratios)), np.array(poison_ratios)*10)
    plt.xlabel('Poison Ratio: PR (%)')
    plt.ylabel('Attack Success Rate: ASR')
    plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}{suffix}.pdf')
    
    return


def hyperparameter_pr_figure_defense(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_pr_asnpca', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    poison_ratios = [0.01, 0.03, 0.05, 0.1, 0.3, 0.4, 0.5]
    backdoor_types = [f'{backdoor_prefix}_backdoor_{pr}' for pr in poison_ratios]
    
    keys = ['ca', 'pa']
    
    results_arr_dc = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        ['snpca_id'], 
        keys=keys,
        results_path_local=results_path_local
    )
    results_arr_mr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        ['snpca_ood'], 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.append(results_arr_dc, results_arr_mr, axis=2)
    
    figs = []
    
    fig = plt.figure(figsize=(5, 2.5))
    for s, defense_type in enumerate(['snpca_id', 'snpca_ood']):
        plt.plot(results_arr[0, :, s, 0], marker='o', markerfacecolor='none', label=f'CA: {nicer_names_for_figures[defense_type]}')
        plt.plot(results_arr[0, :, s, 1], marker='x', label=f'ASR: {nicer_names_for_figures[defense_type]}')
    plt.xticks(np.arange(len(poison_ratios)), np.array(poison_ratios)*10)
    plt.xlabel('Poison Ratio: PR (%)')
    plt.ylabel('Percentage')
    plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def hyperparameter_pr_figure_defense(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_pr_asnpca', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    poison_ratios = [0.01, 0.03, 0.05, 0.1, 0.3, 0.4, 0.5]
    backdoor_types = [f'{backdoor_prefix}_backdoor_{pr}' for pr in poison_ratios]
    
    keys = ['ca', 'pa']
    
    results_arr_dc = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        ['snpca_id'], 
        keys=keys,
        results_path_local=results_path_local
    )
    results_arr_mr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        ['snpca_ood'], 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.append(results_arr_dc, results_arr_mr, axis=2)
    
    linestyles = [None, 'dashed']
    figs = []
    fig, ax1 = plt.subplots(figsize=(5, 2.5))
    ax2 = ax1.twinx()
    for s, defense_type in enumerate(['snpca_id', 'snpca_ood']):
        ax1.plot(results_arr[0, :, s, 0], marker='o', markerfacecolor='none', linestyle=linestyles[s], color='blue', label=f'{nicer_names_for_figures[defense_type]}')
        ax2.plot(results_arr[0, :, s, 1], marker='x', markerfacecolor='none', linestyle=linestyles[s], color='red', label=f'{nicer_names_for_figures[defense_type]}')
    plt.xticks(np.arange(len(poison_ratios)), np.array(poison_ratios)*10)
    ax1.set_xlabel('Poison Ratio: PR (%)')
    ax1.set_ylabel('Clean Accuracy: CA', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel('Attack Success Rate: ASR', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.legend(ncol=2, loc='lower right')
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def hyperparameter_repititions(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_repititions', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    repititions = [1, 10, 50, 100, 500, 1000]
    backdoor_types = [f'{backdoor_prefix}_backdoor_0.3']
    
    defense_types_id = [f'snpca_id_{'(subset_population=None)_' if rp==1 else ''}(repititions={rp})' for rp in repititions]
    defense_types_ood = [f'snpca_ood_{'(subset_population=None)_' if rp==1 else ''}(repititions={rp})' for rp in repititions]
    
    keys = ['ca', 'pa']
    
    results_arr_dc = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_id, 
        keys=keys,
        results_path_local=results_path_local
    )
    results_arr_mr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_ood, 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.append(results_arr_dc, results_arr_mr, axis=0)
    print(results_arr.shape)
    
    figs = []
    
    fig = plt.figure(figsize=(5, 2.5))
    for s, defense_type in enumerate(['snpca_id', 'snpca_ood']):
        # nicer_name = ''
        # if 'snpca_id' in defense_type: nicer_name = nicer_names_for_figures['snpca_id']
        # elif 'snpca_ood' in defense_type: nicer_name = nicer_names_for_figures['snpca_ood'] 
        # # else: assert False
        
        plt.plot(results_arr[s, 0, :, 0], marker='o', markerfacecolor='none', label=f'CA: {nicer_names_for_figures[defense_type]}')
        plt.plot(results_arr[s, 0, :, 1], marker='x', markerfacecolor='none', label=f'ASR: {nicer_names_for_figures[defense_type]}')
    
    plt.xticks(np.arange(len(repititions)), repititions)
    plt.xlabel('Smoothing Repititions: $L$')
    plt.ylabel('Percentage')
    plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def hyperparameter_subsetpopulation(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_subset_population', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    population = [5, 10, 20, 50, 100]
    backdoor_types = [f'{backdoor_prefix}_backdoor_0.3']
    
    defense_types_id = [f'snpca_id_(subset_population={rp})' for rp in population]
    defense_types_ood = [f'snpca_ood_(subset_population={rp})' for rp in population]
    
    keys = ['ca', 'pa']
    
    results_arr_dc = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_id, 
        keys=keys,
        results_path_local=results_path_local
    )
    results_arr_mr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_ood, 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.append(results_arr_dc, results_arr_mr, axis=0)
    print(results_arr.shape)
    
    figs = []
    
    fig = plt.figure(figsize=(5, 2.5))
    for s, defense_type in enumerate(['snpca_id', 'snpca_ood']):
        # nicer_name = ''
        # if 'snpca_id' in defense_type: nicer_name = nicer_names_for_figures['snpca_id']
        # elif 'snpca_ood' in defense_type: nicer_name = nicer_names_for_figures['snpca_ood'] 
        # # else: assert False
        
        plt.plot(results_arr[s, 0, :, 0], marker='o', markerfacecolor='none', label=f'CA: {nicer_names_for_figures[defense_type]}')
        plt.plot(results_arr[s, 0, :, 1], marker='x', markerfacecolor='none', label=f'ASR: {nicer_names_for_figures[defense_type]}')
    
    plt.xticks(np.arange(len(population)), population)
    plt.xlabel('Sampled Subset Population: $d_*^{a(w)}$')
    plt.ylabel('Percentage')
    plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def __hyperparameter_masking_ratio(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_masking_ratio', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    masking_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    backdoor_types = [f'{backdoor_prefix}_backdoor_0.3']
    defense_types_ood = [f'snpca_ood_(mask_ratio={rp})' for rp in masking_ratios]
    
    keys = ['ca', 'pa']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_ood, 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    
    figs = []
    
    fig = plt.figure(figsize=(5, 2.5))
    for s, defense_type in enumerate(['snpca_ood']):
        # nicer_name = ''
        # if 'snpca_id' in defense_type: nicer_name = nicer_names_for_figures['snpca_id']
        # elif 'snpca_ood' in defense_type: nicer_name = nicer_names_for_figures['snpca_ood'] 
        # # else: assert False
        
        plt.plot(results_arr[s, 0, :, 0], marker='o', markerfacecolor='none', label=f'CA: {nicer_names_for_figures[defense_type]}')
        plt.plot(results_arr[s, 0, :, 1], marker='x', markerfacecolor='none', label=f'ASR: {nicer_names_for_figures[defense_type]}')
    
    plt.xticks(np.arange(len(masking_ratios)), masking_ratios)
    plt.xlabel(f'Masking Ratios: $m$ (Step 3 of {nicer_names_for_figures['snpca_ood']})')
    plt.ylabel('Percentage')
    plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def hyperparameter_masking_ratio(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_masking_ratio', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    masking_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    backdoor_types = [f'{backdoor_prefix}_backdoor_0.3']
    defense_types_ood = [f'snpca_ood_(mask_ratio={rp})' for rp in masking_ratios]
    
    keys = ['ca', 'pa']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_ood, 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    print(results_arr.shape)
    
    figs = []
    
    fig, ax1 = plt.subplots(figsize=(5, 2.5))
    ax2 = ax1.twinx()
    for s, defense_type in enumerate(['snpca_ood']):
        # nicer_name = ''
        # if 'snpca_id' in defense_type: nicer_name = nicer_names_for_figures['snpca_id']
        # elif 'snpca_ood' in defense_type: nicer_name = nicer_names_for_figures['snpca_ood'] 
        # # else: assert False
        
        ax1.plot(results_arr[s, 0, :, 0], marker='o', markerfacecolor='none', color='blue', label=f'{nicer_names_for_figures[defense_type]}')
        ax2.plot(results_arr[s, 0, :, 1], marker='x', markerfacecolor='none', color='red', label=f'{nicer_names_for_figures[defense_type]}')
    
    plt.xticks(np.arange(len(masking_ratios)), masking_ratios)
    ax1.set_xlabel(f'Masking Ratios: $m$ (Step 3 of {nicer_names_for_figures['snpca_ood']})')
    ax1.set_ylabel('Clean Accuracy: CA', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel('Attack Success Rate: ASR', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    # plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def __hyperparameter_patch_size(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_patch_size', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    patch_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    backdoor_types = [f'{backdoor_prefix}_backdoor_0.3']
    defense_types_ood = [f'snpca_ood_(patch_ratio={rp})' for rp in patch_ratios]
    
    keys = ['ca', 'pa']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_ood, 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    print(results_arr.shape)
    
    figs = []
    
    fig = plt.figure(figsize=(5, 2.5))
    for s, defense_type in enumerate(['snpca_ood']):
        # nicer_name = ''
        # if 'snpca_id' in defense_type: nicer_name = nicer_names_for_figures['snpca_id']
        # elif 'snpca_ood' in defense_type: nicer_name = nicer_names_for_figures['snpca_ood'] 
        # # else: assert False
        
        plt.plot(results_arr[s, 0, :, 0], marker='o', markerfacecolor='none', label=f'CA: {nicer_names_for_figures[defense_type]}')
        plt.plot(results_arr[s, 0, :, 1], marker='x', markerfacecolor='none', label=f'ASR: {nicer_names_for_figures[defense_type]}')
    
    plt.xticks(np.arange(len(patch_ratios)), patch_ratios)
    plt.xlabel(f'Patch Size (Step 2 of {nicer_names_for_figures['snpca_ood']})')
    plt.ylabel('Percentage')
    plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def hyperparameter_patch_size(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_patch_size', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    patch_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    backdoor_types = [f'{backdoor_prefix}_backdoor_0.3']
    defense_types_ood = [f'snpca_ood_(patch_ratio={rp})' for rp in patch_ratios]
    
    keys = ['ca', 'pa']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_ood, 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    print(results_arr.shape)
    
    figs = []
    
    fig, ax1 = plt.subplots(figsize=(5, 2.5))
    ax2 = ax1.twinx()
    for s, defense_type in enumerate(['snpca_ood']):
        # nicer_name = ''
        # if 'snpca_id' in defense_type: nicer_name = nicer_names_for_figures['snpca_id']
        # elif 'snpca_ood' in defense_type: nicer_name = nicer_names_for_figures['snpca_ood'] 
        # # else: assert False
        
        ax1.plot(results_arr[s, 0, :, 0], marker='o', markerfacecolor='none', color='blue', label=f'{nicer_names_for_figures[defense_type]}')
        ax2.plot(results_arr[s, 0, :, 1], marker='x', markerfacecolor='none', color='red', label=f'{nicer_names_for_figures[defense_type]}')
    
    plt.xticks(np.arange(len(patch_ratios)), patch_ratios)
    ax1.set_xlabel(f'Patch Size (Step 2 of {nicer_names_for_figures['snpca_ood']})')
    ax1.set_ylabel('Clean Accuracy: CA', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel('Attack Success Rate: ASR', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    # plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def __hyperparameter_available_samples(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_available_samples', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    num_samples = [5, 10, 20, 30, 40, 50]
    # num_samples = [10, 20, 30, 40]
    backdoor_types = [f'{backdoor_prefix}_backdoor_0.3']
    defense_types_ood = [f'snpca_ood_(accessible_samples={ns})' for ns in num_samples]
    # defense_types_ood = [f'snpca_ood_(accessible_samples={ns})' if ns!=10 else 'snpca_ood' for ns in num_samples]
    
    keys = ['ca', 'pa']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_ood, 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    print(results_arr.shape)
    
    figs = []
    
    fig = plt.figure(figsize=(5, 2.5))
    for s, defense_type in enumerate(['snpca_ood']):
        # nicer_name = ''
        # if 'snpca_id' in defense_type: nicer_name = nicer_names_for_figures['snpca_id']
        # elif 'snpca_ood' in defense_type: nicer_name = nicer_names_for_figures['snpca_ood'] 
        # # else: assert False
        
        plt.plot(results_arr[s, 0, :, 0], marker='o', markerfacecolor='none', label=f'CA: {nicer_names_for_figures[defense_type]}')
        plt.plot(results_arr[s, 0, :, 1], marker='x', markerfacecolor='none', label=f'ASR: {nicer_names_for_figures[defense_type]}')
    
    plt.xticks(np.arange(len(num_samples)), num_samples)
    plt.xlabel(f'Patch Size (Step 2 of {nicer_names_for_figures['snpca_ood']})')
    plt.ylabel('Percentage')
    plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def hyperparameter_available_samples(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_available_samples', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    num_samples = [5, 10, 20, 30, 40, 50]
    # num_samples = [10, 20, 30, 40]
    backdoor_types = [f'{backdoor_prefix}_backdoor_0.3']
    defense_types_ood = [f'snpca_ood_(accessible_samples={ns})' if ns!=10 else 'snpca_ood' for ns in num_samples]
    defense_types_ood = [f'snpca_ood_(accessible_samples={ns})' for ns in num_samples]
    
    keys = ['ca', 'pa']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_ood, 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    print(results_arr.shape)
    
    figs = []
    
    fig, ax1 = plt.subplots(figsize=(5, 2.5))
    ax2 = ax1.twinx()
    for s, defense_type in enumerate(['snpca_ood']):
        # nicer_name = ''
        # if 'snpca_id' in defense_type: nicer_name = nicer_names_for_figures['snpca_id']
        # elif 'snpca_ood' in defense_type: nicer_name = nicer_names_for_figures['snpca_ood'] 
        # # else: assert False
        
        ax1.plot(results_arr[s, 0, :, 0], marker='o', markerfacecolor='none', color='blue', label=f'{nicer_names_for_figures[defense_type]}')
        ax2.plot(results_arr[s, 0, :, 1], marker='x', markerfacecolor='none', color='red', label=f'{nicer_names_for_figures[defense_type]}')
    
    plt.xticks(np.arange(len(num_samples)), num_samples)
    ax1.set_xlabel(f'# Available Clean Samples (Step 4 of {nicer_names_for_figures['snpca_ood']})')
    ax1.set_ylabel('Clean Accuracy: CA', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel('Attack Success Rate: ASR', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    # plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def __hyperparameter_adversarial_epsilon(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_epsilon', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    population = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    backdoor_types = [f'{backdoor_prefix}_backdoor_0.3']
    
    defense_types_id = [f'snpca_id_(adversarial_epsilon={rp})' if rp!=0.5 else 'snpca_id' for rp in population]
    defense_types_ood = [f'snpca_ood_(adversarial_epsilon={rp})' if rp!=0.5 else 'snpca_ood' for rp in population]
    
    keys = ['ca', 'pa']
    
    results_arr_dc = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_id, 
        keys=keys,
        results_path_local=results_path_local
    )
    results_arr_mr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_ood, 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.append(results_arr_dc, results_arr_mr, axis=0)
    print(results_arr.shape)
    
    figs = []
    
    fig = plt.figure(figsize=(5, 2.5))
    for s, defense_type in enumerate(['snpca_id', 'snpca_ood']):
        # nicer_name = ''
        # if 'snpca_id' in defense_type: nicer_name = nicer_names_for_figures['snpca_id']
        # elif 'snpca_ood' in defense_type: nicer_name = nicer_names_for_figures['snpca_ood'] 
        # # else: assert False
        
        plt.plot(results_arr[s, 0, :, 0], marker='o', markerfacecolor='none', label=f'CA: {nicer_names_for_figures[defense_type]}')
        plt.plot(results_arr[s, 0, :, 1], marker='x', markerfacecolor='none', label=f'ASR: {nicer_names_for_figures[defense_type]}')
    
    plt.xticks(np.arange(len(population)), population)
    plt.xlabel('Adversarial Espilon: $\\epsilon$')
    plt.ylabel('Percentage')
    plt.legend()
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


def hyperparameter_adversarial_epsilon(dataset_names, results_path_local: str, backdoor_prefix: str='simple', figure_name: str='hyperparameter_epsilon', save_fig: bool=False):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    population = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    backdoor_types = [f'{backdoor_prefix}_backdoor_0.3']
    
    defense_types_id = [f'snpca_id_(adversarial_epsilon={rp})' if rp!=0.5 else 'snpca_id' for rp in population]
    defense_types_ood = [f'snpca_ood_(adversarial_epsilon={rp})' if rp!=0.5 else 'snpca_ood' for rp in population]
    
    keys = ['ca', 'pa']
    
    results_arr_dc = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_id, 
        keys=keys,
        results_path_local=results_path_local
    )
    results_arr_mr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types_ood, 
        keys=[f'{key}_MR' for key in keys],
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.append(results_arr_dc, results_arr_mr, axis=0)
    print(results_arr.shape)
    
    figs = []
    
    markers = ['o', 'x']; linestyles = [None, 'dashed']
    fig, ax1 = plt.subplots(figsize=(5, 2.5))
    ax2 = ax1.twinx()
    for s, defense_type in enumerate(['snpca_id', 'snpca_ood']):
        # nicer_name = ''
        # if 'snpca_id' in defense_type: nicer_name = nicer_names_for_figures['snpca_id']
        # elif 'snpca_ood' in defense_type: nicer_name = nicer_names_for_figures['snpca_ood'] 
        # # else: assert False
        
        ax1.plot(results_arr[s, 0, :, 0], marker=markers[s], markerfacecolor='none', color='blue', linestyle=linestyles[s], label=f'{nicer_names_for_figures[defense_type]}')
        ax2.plot(results_arr[s, 0, :, 1], marker=markers[s], markerfacecolor='none', color='red', linestyle=linestyles[s], label=f'{nicer_names_for_figures[defense_type]}')
    
    plt.xticks(np.arange(len(population)), population)
    ax1.set_xlabel('Adversarial Espilon: $\\epsilon$')
    ax1.set_ylabel('Clean Accuracy: CA', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel('Attack Success Rate: ASR', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.legend(loc='center right')
    plt.tight_layout()
    figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{backdoor_prefix}_{figure_name}.pdf')
    
    return


