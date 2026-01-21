from ..config import results_path

from ..visual_utils.make_tables import comparison_with_sota_dc, comparison_with_sota_mr, comparison_with_sota_mf



def generate_sota_analysis_tables_dc():
    
    identifier_string = 'DC'
    fn_to_call = comparison_with_sota_dc
    
    print(f'\n\nResults MNIST {identifier_string}:')
    print(fn_to_call('mnist', results_path))
    
    print(f'\n\nResults CIFAR-10 {identifier_string}:')
    print(fn_to_call('cifar10', results_path))
    
    print(f'\n\nResults CIFAR-10 Convnext {identifier_string}:')
    print(fn_to_call('cifar10_convnext', results_path))
    
    print(f'\n\nResults CIFAR-100 {identifier_string}:')
    print(fn_to_call('cifar100', results_path))
    
    print(f'\n\nResults CIFAR-100 Convnext {identifier_string}:')
    print(fn_to_call('cifar100_convnext', results_path))
    
    # Leave the rest commented for now
    # print(f'\n\nResults GTSRB {identifier_string}:')
    # print(comparison_with_sota('gtsrb', results_path))
    
    return


def generate_sota_analysis_tables_mr():
    
    identifier_string = 'MR'
    fn_to_call = comparison_with_sota_mr
    
    print(f'\n\nResults MNIST {identifier_string}:')
    print(fn_to_call('mnist', results_path))
    
    print(f'\n\nResults CIFAR-10 {identifier_string}:')
    print(fn_to_call('cifar10', results_path))
    
    print(f'\n\nResults CIFAR-10 Convnext {identifier_string}:')
    print(fn_to_call('cifar10_convnext', results_path))
    
    print(f'\n\nResults CIFAR-100 {identifier_string}:')
    print(fn_to_call('cifar100', results_path))
    
    print(f'\n\nResults CIFAR-100 Convnext {identifier_string}:')
    print(fn_to_call('cifar100_convnext', results_path))
    
    # Leave the rest commented for now
    # print(f'\n\nResults GTSRB {identifier_string}:')
    # print(comparison_with_sota('gtsrb', results_path))
    
    return


def generate_sota_analysis_tables_mf():
    
    identifier_string = 'MF'
    fn_to_call = comparison_with_sota_mf
    
    print(f'\n\nResults MNIST {identifier_string}:')
    print(fn_to_call('cifar10_mnist', results_path))
    
    print(f'\n\nResults CIFAR-10 {identifier_string}:')
    print(fn_to_call('cifar100_cifar10', results_path))
    
    # print(f'\n\nResults CIFAR-10 Convnext {identifier_string}:')
    # print(fn_to_call(['cifar10_convnext'], results_path))
    
    # print(f'\n\nResults CIFAR-100 {identifier_string}:')
    # print(fn_to_call(['cifar100'], results_path))
    
    # print(f'\n\nResults CIFAR-100 Convnext {identifier_string}:')
    # print(fn_to_call(['cifar100_convnext'], results_path))
    
    # # print(f'\n\nResults GTSRB {identifier_string}:')
    # # print(comparison_with_sota(['gtsrb'], results_path))
    
    return


def main(scenario: str='DC'):
    
    if scenario == 'DC':
        print('\n\n\nResults for the DC: Data collection scenario.')
        generate_sota_analysis_tables_dc()
    
    elif scenario == 'MR':
        print('\n\n\nResults for the MR: Model reuse scenario.')
        generate_sota_analysis_tables_mr()
    
    elif scenario == 'MF':
        print('\n\n\nResults for the MF: Model finetune scenario.')
        generate_sota_analysis_tables_mf()
    
    return

