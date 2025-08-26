from ..config import results_path

from ..visual_utils.make_tables import comparison_with_sota_dc, comparison_with_sota_mr, comparison_with_sota_mf



def generate_sota_analysis_tables_dc():
    
    print('\n\nTable 1 of the paper here:')
    print(comparison_with_sota_dc(['mnist'], results_path))
    
    # print('\n\nTable 2 of the paper here:')
    print(comparison_with_sota_dc(['cifar10'], results_path))
    
    # print('\n\nTable 3 of the paper here:')
    # print(comparison_with_sota(['gtsrb'], results_path))
    
    # print('\n\nTable 3 of the paper here:')
    print(comparison_with_sota_dc(['cifar100'], results_path))
    
    return


def generate_sota_analysis_tables_mr():
    
    # print('\n\nTable 1 of the paper here:')
    print(comparison_with_sota_mr(['mnist'], results_path))
    
    # print('\n\nTable 2 of the paper here:')
    print(comparison_with_sota_mr(['cifar10'], results_path))
    
    # print('\n\nTable 3 of the paper here:')
    # print(comparison_with_sota_tm2(['gtsrb'], results_path))
    
    # print('\n\nTable 3 of the paper here:')
    print(comparison_with_sota_mr(['cifar100'], results_path))
    
    return


def generate_sota_analysis_tables_mf():
    
    # print('\n\nTable 1 of the paper here:')
    print(comparison_with_sota_mf(['mnist'], results_path))
    
    # print('\n\nTable 2 of the paper here:')
    print(comparison_with_sota_mf(['cifar10'], results_path))
    
    # print('\n\nTable 3 of the paper here:')
    # print(comparison_with_sota_mr(['gtsrb'], results_path))
    
    # print('\n\nTable 3 of the paper here:')
    print(comparison_with_sota_mf(['cifar100'], results_path))
    
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

