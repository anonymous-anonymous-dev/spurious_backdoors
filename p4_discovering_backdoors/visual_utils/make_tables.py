import numpy as np
import matplotlib.pyplot as plt


from .loading import load_results_from_settings



nicer_names = {
    'vanilla': 'No Defense',
    'strip': 'STRIP',
    'activation_clustering': 'AC',
    'spectral_signatures': 'SS',
    'mdtd': 'MDTD', 
    'zero_shot_purification': 'ZIP',
    'snpca_id': '\\textbf{\\dfdata{}}',
    'snpca_ood': '\\textbf{\\dfnodata{}}',
    
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


def __comparison_with_sota_dc(dataset_names, results_path_local: str):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    backdoor_types = [
        'simple_backdoor_0',
        
        # 'simple_backdoor_0.01_global_poisoning',
        # 'invisible_backdoor_0.01_global_poisoning',
        # # 'reflection_backdoor_0.01_global_poisoning',
        # # # 'clean_label_backdoor_0.01_global_poisoning',
        # # 'wanet_backdoor_0.01_global_poisoning',
        # 'horizontal_backdoor_0.01_global_poisoning',
        
        'simple_backdoor_0.03_global_poisoning',
        'invisible_backdoor_0.03_global_poisoning',
        # 'reflection_backdoor_0.03_global_poisoning',
        # # 'clean_label_backdoor_0.03_global_poisoning',
        # 'wanet_backdoor_0.03_global_poisoning',
        'horizontal_backdoor_0.03_global_poisoning',
        
    ]
    
    defense_type_theirs = [
        'vanilla',
        'strip', 'spectral_signatures', 'activation_clustering', 'mdtd', 'zero_shot_purification',
    ]
    defense_type_ours = [
        'snpca_id', 
        # 'snpca_ood',
    ]
    defense_types = defense_type_theirs + defense_type_ours
    
    keys = ['ca', 'pa']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.round(results_arr, decimals=2)
    _results_arr = results_arr.copy()
    best_accs = np.max(_results_arr[:,:,1:,0], axis=(2)); _results_arr[_results_arr<0] = 3
    best_asrs = np.min(_results_arr[:,:,1:,1], axis=(2))
    # print(results_arr.shape, best_accs.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(backdoor_types))
    
    table_string = ''
    for s, defense_type in enumerate(defense_types):
        
        # if defense_type == defense_type_ours[0]:
        #     table_string += '\\hline\n'
        
        table_string += '& {}'.format(nicer_names[defense_type])
        
        for d, dataset_name in enumerate(dataset_names):
            for b, backdoor_type in enumerate(backdoor_types):
                
                result_ = results_arr[d, b, s]
                
                acc_ = f'{result_[0]:.2f}' if result_[0] >= 0 else '-'
                asr_ = f'{result_[1]:.2f}' if result_[1] >= 0 else '-'
                if (acc_!='-') & (s!=0):
                    acc_ = '\\textbf{' + acc_ + '}' if result_[0]==best_accs[d,b] else acc_
                    asr_ = '\\underline{' + asr_ + '}' if result_[1]==best_asrs[d,b] else asr_
                table_string += f' & {acc_}({asr_})'
                # table_string += ' & {}({})'.format(
                #     '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                #     '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                # )
        
            table_string += '\n'
        table_string += '\\\\\n'
    table_string += '\\hline\n'
    
    return table_string


def comparison_with_sota_dc(dataset_name, results_path_local: str):
    
    dataset_names = [dataset_name]
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    backdoor_types = [
        'simple_backdoor_0',
        
        # 'simple_backdoor_0.1',
        # 'invisible_backdoor_0.1',
        # 'clean_label_backdoor_0.1',
        # 'reflection_backdoor_0.1',
        # 'wanet_backdoor_0.1',
        
        'simple_backdoor_0.3',
        'invisible_backdoor_0.3',
        'clean_label_backdoor_0.3',
        'reflection_backdoor_0.3',
        'wanet_backdoor_0.3', # if not 'cifar100' in dataset_name else 'wanet_backdoor_10.0',
        'horizontal_backdoor_0.3',
        
    ]
    
    defense_type_theirs = [
        'vanilla',
        'strip', 'spectral_signatures', 'activation_clustering', 'mdtd', 'zero_shot_purification',
    ]
    defense_type_ours = [
        'snpca_id', 
        # 'snpca_ood',
    ]
    defense_types = defense_type_theirs + defense_type_ours
    
    keys = ['ca', 'pa']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.round(results_arr, decimals=2)
    _results_arr = results_arr.copy()
    best_accs = np.max(_results_arr[:,:,1:,0], axis=(2)); _results_arr[_results_arr<0] = 3
    best_asrs = np.min(_results_arr[:,:,1:,1], axis=(2))
    # print('|c|' + 'c|'*len(dataset_names)*len(backdoor_types))
    
    table_string = ''
    for s, defense_type in enumerate(defense_types):
        
        # if defense_type == defense_type_ours[0]:
        #     table_string += '\\hline\n'
        
        table_string += '& {}'.format(nicer_names[defense_type])
        
        for d, dataset_name in enumerate(dataset_names):
            for b, backdoor_type in enumerate(backdoor_types):
                
                result_ = results_arr[d, b, s]
                
                acc_ = f'{result_[0]:.2f}' if result_[0] >= 0 else '-'
                asr_ = f'{result_[1]:.2f}' if result_[1] >= 0 else '-'
                if (acc_!='-') & (s!=0):
                    acc_ = '\\textbf{' + acc_ + '}' if result_[0]==best_accs[d,b] else acc_
                    asr_ = '\\underline{' + asr_ + '}' if result_[1]==best_asrs[d,b] else asr_
                table_string += f' & {acc_}({asr_})'
                # table_string += ' & {}({})'.format(
                #     '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                #     '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                # )
        
            table_string += '\n'
        table_string += '\\\\\n'
    table_string += '\\hline\n'
    
    return table_string


def comparison_with_sota_mr(dataset_name: str, results_path_local: str):
    
    dataset_names = [dataset_name]
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    backdoor_types = [
        'simple_backdoor_0',
        
        # 'simple_backdoor_0.1',
        # 'invisible_backdoor_0.1',
        # 'clean_label_backdoor_0.1',
        # 'reflection_backdoor_0.1',
        # 'wanet_backdoor_0.1',
        
        'simple_backdoor_0.3',
        'invisible_backdoor_0.3',
        'clean_label_backdoor_0.3',
        'reflection_backdoor_1.0' if 'cifar100' in dataset_name else 'reflection_backdoor_0.3',
        'wanet_backdoor_10.0' if 'cifar100' in dataset_name else 'wanet_backdoor_1.0',
        'horizontal_backdoor_3.0' if 'cifar100' in dataset_name else 'horizontal_backdoor_0.3',
        
    ]
    
    defense_type_theirs = [
        'vanilla',
        'strip', 'spectral_signatures', 'activation_clustering', 'mdtd', 'zero_shot_purification',
    ]
    defense_type_ours = [
        # 'snpca_id', 
        'snpca_ood',
    ]
    defense_types = defense_type_theirs + defense_type_ours
    
    keys = ['ca_MR', 'pa_MR']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.round(results_arr, decimals=2)
    _results_arr = results_arr.copy()
    best_accs = np.max(_results_arr[:,:,1:,0], axis=(2)); _results_arr[_results_arr<0] = 3
    best_asrs = np.min(_results_arr[:,:,1:,1], axis=(2))
    # print('|c|' + 'c|'*len(dataset_names)*len(backdoor_types))
    
    table_string = ''
    for s, defense_type in enumerate(defense_types):
        
        # if defense_type == defense_type_ours[0]:
        #     table_string += '\\hline\n'
        
        table_string += '& {}'.format(nicer_names[defense_type])
        
        for d, dataset_name in enumerate(dataset_names):
            for b, backdoor_type in enumerate(backdoor_types):
                
                result_ = results_arr[d, b, s]
                acc_ = f'{result_[0]:.2f}' if result_[0] >= 0 else '-'
                asr_ = f'{result_[1]:.2f}' if result_[1] >= 0 else '-'
                if (acc_!='-') & (s!=0):
                    acc_ = '\\textbf{' + acc_ + '}' if float(acc_)>=best_accs[d,b] else acc_
                    asr_ = '\\underline{' + asr_ + '}' if float(asr_)<=best_asrs[d,b] else asr_
                table_string += f' & {acc_}({asr_})'
        
            table_string += '\n'
        table_string += '\\\\\n'
    table_string += '\\hline\n'
    
    return table_string


def comparison_with_sota_mf(dataset_name, results_path_local: str):
    
    dataset_names = [dataset_name]
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    backdoor_types = [
        'simple_backdoor_0',
        
        # 'simple_backdoor_0.1',
        # 'invisible_backdoor_0.1',
        # 'clean_label_backdoor_0.1',
        # 'reflection_backdoor_0.1',
        # 'wanet_backdoor_0.1',
        
        'simple_backdoor_0.3',
        'invisible_backdoor_0.3',
        'clean_label_backdoor_0.3',
        'reflection_backdoor_1.0' if 'cifar100' in dataset_name else 'reflection_backdoor_0.3',
        'wanet_backdoor_10.0' if 'cifar100' in dataset_name else 'wanet_backdoor_1.0',
        'horizontal_backdoor_3.0' if 'cifar100' in dataset_name else 'horizontal_backdoor_0.3',
        
    ]
    
    defense_type_theirs = [
        'vanilla',
        'strip', 'spectral_signatures', 'activation_clustering', 'mdtd', 'zero_shot_purification',
    ]
    defense_type_ours = [
        # 'snpca_id', 
        'snpca_ood',
    ]
    defense_types = defense_type_theirs + defense_type_ours
    
    keys = ['ca_MF', 'pa_MF']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        backdoor_types, 
        defense_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    results_arr = np.round(results_arr, decimals=2)
    _results_arr = results_arr.copy()
    best_accs = np.max(_results_arr[:,:,1:,0], axis=(2)); _results_arr[_results_arr<0] = 3
    best_asrs = np.min(_results_arr[:,:,1:,1], axis=(2))
    # print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(backdoor_types))
    
    table_string = ''
    for s, defense_type in enumerate(defense_types):
        
        table_string += '& {}'.format(nicer_names[defense_type])
        
        for d, dataset_name in enumerate(dataset_names):
            for b, backdoor_type in enumerate(backdoor_types):
                
                result_ = results_arr[d, b, s]
                acc_ = f'{result_[0]:.2f}' if result_[0] >= 0 else '-'
                asr_ = f'{result_[1]:.2f}' if result_[1] >= 0 else '-'
                if (acc_!='-') & (s!=0):
                    acc_ = '\\textbf{' + acc_ + '}' if result_[0]==best_accs[d,b] else acc_
                    asr_ = '\\underline{' + asr_ + '}' if result_[1]==best_asrs[d,b] else asr_
                table_string += f' & {acc_}({asr_})'
                
            table_string += '\n'
        table_string += '\\\\\n'
    table_string += '\\hline\n'
    
    return table_string


