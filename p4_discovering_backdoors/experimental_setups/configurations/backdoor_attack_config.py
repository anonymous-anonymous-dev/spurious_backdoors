poison_ratio = 0.03
target = 1


# spurious_simple_backdoor_configuration = {
#     'poison_ratio': poison_ratio,
#     'poison_ratio_wrt_class_members': True
# }
simple_backdoor_configuration = {
    'poison_ratio': poison_ratio,
    'poison_ratio_wrt_class_members': True
}
invisible_backdoor_configuration = {
    'poison_ratio': poison_ratio,
    'poison_ratio_wrt_class_members': True
}
reflection_backdoor_configuration = {
    'poison_ratio': poison_ratio,
    'poison_ratio_wrt_class_members': True,
    # 'guassian_kernel_size': 5
}
multi_target_backdoor_configuration = {
    'poison_ratio': poison_ratio,
    'poison_ratio_wrt_class_members': True,
    'num_targets': 4
}
clean_label_backdoor_configuration = {
    'poison_ratio': poison_ratio,
    'poison_ratio_wrt_class_members': True,
    # 'epsilon': 0.1
}
wanet_backdoor_configuration = {
    'poison_ratio': poison_ratio,
    'poison_ratio_wrt_class_members': True,
}
horizontal_class_backdoor_configuration = {
    'cover_ratio': 0.4,
    'poison_ratio': poison_ratio,
    'poison_ratio_wrt_class_members': True,
}


backdoor_configurations = {
    'simple_backdoor': simple_backdoor_configuration,
    'invisible_backdoor': invisible_backdoor_configuration,
    'multitarget_backdoor': multi_target_backdoor_configuration,
    'reflection_backdoor': reflection_backdoor_configuration,
    'clean_label_backdoor': clean_label_backdoor_configuration,
    'wanet_backdoor': wanet_backdoor_configuration,
    'horizontal_backdoor': horizontal_class_backdoor_configuration,
    
    # spuriousity analysis
    'spurious': simple_backdoor_configuration,
    'spurious_sunset': simple_backdoor_configuration,
    'spurious_dog': simple_backdoor_configuration,
    
}


backdoor_attacks_configured = {
    
    'spurious_0.1': {'type': 'spurious', 'poison_ratio': 0.1},
    'spurious_0.3': {'type': 'spurious', 'poison_ratio': 0.3},
    
    'spurious_dog_0.1': {'type': 'spurious_dog', 'poison_ratio': 0.1, 'target': 5},
    'spurious_sun_0.1': {'type': 'spurious_sunset', 'poison_ratio': 0.1, 'target': 0},
    
    'simple_backdoor': {'type': 'simple_backdoor'},
    'simple_backdoor_0': {'type': 'simple_backdoor', 'poison_ratio': 0.},
    'simple_backdoor_0.01': {'type': 'simple_backdoor', 'poison_ratio': 0.01},
    'simple_backdoor_0.03': {'type': 'simple_backdoor', 'poison_ratio': 0.03},
    'simple_backdoor_0.05': {'type': 'simple_backdoor', 'poison_ratio': 0.05},
    'simple_backdoor_0.1': {'type': 'simple_backdoor', 'poison_ratio': 0.1},
    'simple_backdoor_0.2': {'type': 'simple_backdoor', 'poison_ratio': 0.2},
    'simple_backdoor_0.3': {'type': 'simple_backdoor', 'poison_ratio': 0.3},
    'simple_backdoor_0.4': {'type': 'simple_backdoor', 'poison_ratio': 0.4},
    'simple_backdoor_0.5': {'type': 'simple_backdoor', 'poison_ratio': 0.5},
    
    'invisible_backdoor': {'type': 'invisible_backdoor'},
    'invisible_backdoor_0': {'type': 'invisible_backdoor', 'poison_ratio': 0.},
    'invisible_backdoor_0.01': {'type': 'invisible_backdoor', 'poison_ratio': 0.01},
    'invisible_backdoor_0.03': {'type': 'invisible_backdoor', 'poison_ratio': 0.03},
    'invisible_backdoor_0.05': {'type': 'invisible_backdoor', 'poison_ratio': 0.05},
    'invisible_backdoor_0.1': {'type': 'invisible_backdoor', 'poison_ratio': 0.1},
    'invisible_backdoor_0.3': {'type': 'invisible_backdoor', 'poison_ratio': 0.3},
    'invisible_backdoor_0.4': {'type': 'invisible_backdoor', 'poison_ratio': 0.4},
    'invisible_backdoor_0.5': {'type': 'invisible_backdoor', 'poison_ratio': 0.5},
    
    'multitarget_backdoor': {'type': 'multitarget_backdoor'},
    'multitarget_backdoor_0': {'type': 'multitarget_backdoor', 'poison_ratio': 0.},
    'multitarget_backdoor_0.01': {'type': 'multitarget_backdoor', 'poison_ratio': 0.01},
    'multitarget_backdoor_0.03': {'type': 'multitarget_backdoor', 'poison_ratio': 0.03},
    'multitarget_backdoor_0.05': {'type': 'multitarget_backdoor', 'poison_ratio': 0.05},
    'multitarget_backdoor_0.1': {'type': 'multitarget_backdoor', 'poison_ratio': 0.1},
    'multitarget_backdoor_0.3': {'type': 'multitarget_backdoor', 'poison_ratio': 0.3},
    'multitarget_backdoor_0.4': {'type': 'multitarget_backdoor', 'poison_ratio': 0.4},
    'multitarget_backdoor_0.5': {'type': 'multitarget_backdoor', 'poison_ratio': 0.5},
    
    'reflection_backdoor': {'type': 'reflection_backdoor'},
    'reflection_backdoor_0': {'type': 'reflection_backdoor', 'poison_ratio': 0.},
    'reflection_backdoor_0.01': {'type': 'reflection_backdoor', 'poison_ratio': 0.01},
    'reflection_backdoor_0.03': {'type': 'reflection_backdoor', 'poison_ratio': 0.03},
    'reflection_backdoor_0.05': {'type': 'reflection_backdoor', 'poison_ratio': 0.05},
    'reflection_backdoor_0.1': {'type': 'reflection_backdoor', 'poison_ratio': 0.1},
    'reflection_backdoor_0.2': {'type': 'reflection_backdoor', 'poison_ratio': 0.2},
    'reflection_backdoor_0.3': {'type': 'reflection_backdoor', 'poison_ratio': 0.3},
    'reflection_backdoor_0.4': {'type': 'reflection_backdoor', 'poison_ratio': 0.4},
    'reflection_backdoor_0.5': {'type': 'reflection_backdoor', 'poison_ratio': 0.5},
    
    'persian_cat_refool_0.1': {'type': 'reflection_backdoor', 'poison_ratio': 0.1, 'target': 283},
    'persian_cat_vtba_0.1': {'type': 'simple_backdoor', 'poison_ratio': 0.1, 'target': 283},
    
    'clean_label_backdoor': {'type': 'clean_label_backdoor'},
    'clean_label_backdoor_0': {'type': 'clean_label_backdoor', 'poison_ratio': 0.},
    'clean_label_backdoor_0.01': {'type': 'clean_label_backdoor', 'poison_ratio': 0.01},
    'clean_label_backdoor_0.03': {'type': 'clean_label_backdoor', 'poison_ratio': 0.03},
    'clean_label_backdoor_0.05': {'type': 'clean_label_backdoor', 'poison_ratio': 0.05},
    'clean_label_backdoor_0.1': {'type': 'clean_label_backdoor', 'poison_ratio': 0.1},
    'clean_label_backdoor_0.2': {'type': 'clean_label_backdoor', 'poison_ratio': 0.2},
    'clean_label_backdoor_0.3': {'type': 'clean_label_backdoor', 'poison_ratio': 0.3},
    'clean_label_backdoor_0.4': {'type': 'clean_label_backdoor', 'poison_ratio': 0.4},
    'clean_label_backdoor_0.5': {'type': 'clean_label_backdoor', 'poison_ratio': 0.5},
    
    'wanet_backdoor': {'type': 'wanet_backdoor'},
    'wanet_backdoor_0': {'type': 'wanet_backdoor', 'poison_ratio': 0.},
    'wanet_backdoor_0.01': {'type': 'wanet_backdoor', 'poison_ratio': 0.01},
    'wanet_backdoor_0.03': {'type': 'wanet_backdoor', 'poison_ratio': 0.03},
    'wanet_backdoor_0.05': {'type': 'wanet_backdoor', 'poison_ratio': 0.05},
    'wanet_backdoor_0.1': {'type': 'wanet_backdoor', 'poison_ratio': 0.1},
    'wanet_backdoor_0.2': {'type': 'wanet_backdoor', 'poison_ratio': 0.2},
    'wanet_backdoor_0.3': {'type': 'wanet_backdoor', 'poison_ratio': 0.3},
    'wanet_backdoor_0.4': {'type': 'wanet_backdoor', 'poison_ratio': 0.4},
    'wanet_backdoor_0.5': {'type': 'wanet_backdoor', 'poison_ratio': 0.5},
    
    'horizontal_backdoor': {'type': 'horizontal_backdoor'},
    'horizontal_backdoor_0': {'type': 'horizontal_backdoor', 'poison_ratio': 0.},
    'horizontal_backdoor_0.01': {'type': 'horizontal_backdoor', 'poison_ratio': 0.01},
    'horizontal_backdoor_0.03': {'type': 'horizontal_backdoor', 'poison_ratio': 0.03},
    'horizontal_backdoor_0.05': {'type': 'horizontal_backdoor', 'poison_ratio': 0.05},
    'horizontal_backdoor_0.1': {'type': 'horizontal_backdoor', 'poison_ratio': 0.1},
    'horizontal_backdoor_0.2': {'type': 'horizontal_backdoor', 'poison_ratio': 0.2},
    'horizontal_backdoor_0.3': {'type': 'horizontal_backdoor', 'poison_ratio': 0.3},
    'horizontal_backdoor_0.4': {'type': 'horizontal_backdoor', 'poison_ratio': 0.4},
    'horizontal_backdoor_0.5': {'type': 'horizontal_backdoor', 'poison_ratio': 0.5},
    
    
    
    # =============================================
    # Target class 1
    # =============================================
    '1_simple_backdoor': {'type': 'simple_backdoor', 'target': 1},
    '1_simple_backdoor_0': {'type': 'simple_backdoor', 'poison_ratio': 0., 'target': 1},
    '1_simple_backdoor_0.01': {'type': 'simple_backdoor', 'poison_ratio': 0.01, 'target': 1},
    '1_simple_backdoor_0.03': {'type': 'simple_backdoor', 'poison_ratio': 0.03, 'target': 1},
    '1_simple_backdoor_0.05': {'type': 'simple_backdoor', 'poison_ratio': 0.05, 'target': 1},
    '1_simple_backdoor_0.1': {'type': 'simple_backdoor', 'poison_ratio': 0.1, 'target': 1},
    '1_simple_backdoor_0.2': {'type': 'simple_backdoor', 'poison_ratio': 0.2, 'target': 1},
    '1_simple_backdoor_0.3': {'type': 'simple_backdoor', 'poison_ratio': 0.3, 'target': 1},
    '1_simple_backdoor_0.4': {'type': 'simple_backdoor', 'poison_ratio': 0.4, 'target': 1},
    '1_simple_backdoor_0.5': {'type': 'simple_backdoor', 'poison_ratio': 0.5, 'target': 1},
    
    '1_invisible_backdoor_0.01': {'type': 'invisible_backdoor', 'poison_ratio': 0.01, 'target': 1},
    '1_invisible_backdoor_0.03': {'type': 'invisible_backdoor', 'poison_ratio': 0.03, 'target': 1},
    '1_invisible_backdoor_0.1': {'type': 'invisible_backdoor', 'poison_ratio': 0.1, 'target': 1},
    '1_invisible_backdoor_0.3': {'type': 'invisible_backdoor', 'poison_ratio': 0.3, 'target': 1},
    
    '1_multitarget_backdoor_0.01': {'type': 'multitarget_backdoor', 'poison_ratio': 0.01, 'target': 1},
    '1_multitarget_backdoor_0.03': {'type': 'multitarget_backdoor', 'poison_ratio': 0.03, 'target': 1},
    '1_multitarget_backdoor_0.1': {'type': 'multitarget_backdoor', 'poison_ratio': 0.1, 'target': 1},
    '1_multitarget_backdoor_0.3': {'type': 'multitarget_backdoor', 'poison_ratio': 0.3, 'target': 1},
    
    '1_reflection_backdoor_0.01': {'type': 'reflection_backdoor', 'poison_ratio': 0.01, 'target': 1},
    '1_reflection_backdoor_0.03': {'type': 'reflection_backdoor', 'poison_ratio': 0.03, 'target': 1},
    '1_reflection_backdoor_0.1': {'type': 'reflection_backdoor', 'poison_ratio': 0.1, 'target': 1},
    '1_reflection_backdoor_0.3': {'type': 'reflection_backdoor', 'poison_ratio': 0.3, 'target': 1},
    
    '1_clean_label_backdoor_0.01': {'type': 'clean_label_backdoor', 'poison_ratio': 0.01, 'target': 1},
    '1_clean_label_backdoor_0.03': {'type': 'clean_label_backdoor', 'poison_ratio': 0.03, 'target': 1},
    '1_clean_label_backdoor_0.1': {'type': 'clean_label_backdoor', 'poison_ratio': 0.1, 'target': 1},
    '1_clean_label_backdoor_0.3': {'type': 'clean_label_backdoor', 'poison_ratio': 0.3, 'target': 1},
    
    '1_wanet_backdoor_0.01': {'type': 'wanet_backdoor', 'poison_ratio': 0.01, 'target': 1},
    '1_wanet_backdoor_0.03': {'type': 'wanet_backdoor', 'poison_ratio': 0.03, 'target': 1},
    '1_wanet_backdoor_0.1': {'type': 'wanet_backdoor', 'poison_ratio': 0.1, 'target': 1},
    '1_wanet_backdoor_0.3': {'type': 'wanet_backdoor', 'poison_ratio': 0.3, 'target': 1},
    
    
    # =============================================
    # GLOBAL POISONING
    # =============================================
    'simple_backdoor_global_poisoning': {'type': 'simple_backdoor'},
    'simple_backdoor_0_global_poisoning': {'type': 'simple_backdoor', 'poison_ratio': 0., 'poison_ratio_wrt_class_members': False},
    'simple_backdoor_0.01_global_poisoning': {'type': 'simple_backdoor', 'poison_ratio': 0.01, 'poison_ratio_wrt_class_members': False},
    'simple_backdoor_0.03_global_poisoning': {'type': 'simple_backdoor', 'poison_ratio': 0.03, 'poison_ratio_wrt_class_members': False},
    'simple_backdoor_0.05_global_poisoning': {'type': 'simple_backdoor', 'poison_ratio': 0.05, 'poison_ratio_wrt_class_members': False},
    'simple_backdoor_0.1_global_poisoning': {'type': 'simple_backdoor', 'poison_ratio': 0.1, 'poison_ratio_wrt_class_members': False},
    'simple_backdoor_0.3_global_poisoning': {'type': 'simple_backdoor', 'poison_ratio': 0.3, 'poison_ratio_wrt_class_members': False},
    'simple_backdoor_0.5_global_poisoning': {'type': 'simple_backdoor', 'poison_ratio': 0.5, 'poison_ratio_wrt_class_members': False},
    
    'invisible_backdoor_global_poisoning': {'type': 'invisible_backdoor', 'poison_ratio_wrt_class_members': False},
    'invisible_backdoor_0_global_poisoning': {'type': 'invisible_backdoor', 'poison_ratio': 0., 'poison_ratio_wrt_class_members': False},
    'invisible_backdoor_0.01_global_poisoning': {'type': 'invisible_backdoor', 'poison_ratio': 0.01, 'poison_ratio_wrt_class_members': False},
    'invisible_backdoor_0.03_global_poisoning': {'type': 'invisible_backdoor', 'poison_ratio': 0.03, 'poison_ratio_wrt_class_members': False},
    'invisible_backdoor_0.05_global_poisoning': {'type': 'invisible_backdoor', 'poison_ratio': 0.05, 'poison_ratio_wrt_class_members': False},
    'invisible_backdoor_0.1_global_poisoning': {'type': 'invisible_backdoor', 'poison_ratio': 0.1, 'poison_ratio_wrt_class_members': False},
    'invisible_backdoor_0.3_global_poisoning': {'type': 'invisible_backdoor', 'poison_ratio': 0.3, 'poison_ratio_wrt_class_members': False},
    'invisible_backdoor_0.5_global_poisoning': {'type': 'invisible_backdoor', 'poison_ratio': 0.5, 'poison_ratio_wrt_class_members': False},
    
    'multitarget_backdoor_global_poisoning': {'type': 'multitarget_backdoor', 'poison_ratio_wrt_class_members': False},
    'multitarget_backdoor_0_global_poisoning': {'type': 'multitarget_backdoor', 'poison_ratio': 0., 'poison_ratio_wrt_class_members': False},
    'multitarget_backdoor_0.01_global_poisoning': {'type': 'multitarget_backdoor', 'poison_ratio': 0.01, 'poison_ratio_wrt_class_members': False},
    'multitarget_backdoor_0.03_global_poisoning': {'type': 'multitarget_backdoor', 'poison_ratio': 0.03, 'poison_ratio_wrt_class_members': False},
    'multitarget_backdoor_0.05_global_poisoning': {'type': 'multitarget_backdoor', 'poison_ratio': 0.05, 'poison_ratio_wrt_class_members': False},
    'multitarget_backdoor_0.1_global_poisoning': {'type': 'multitarget_backdoor', 'poison_ratio': 0.1, 'poison_ratio_wrt_class_members': False},
    'multitarget_backdoor_0.3_global_poisoning': {'type': 'multitarget_backdoor', 'poison_ratio': 0.3, 'poison_ratio_wrt_class_members': False},
    
    'reflection_backdoor_global_poisoning': {'type': 'reflection_backdoor', 'poison_ratio_wrt_class_members': False},
    'reflection_backdoor_0_global_poisoning': {'type': 'reflection_backdoor', 'poison_ratio': 0., 'poison_ratio_wrt_class_members': False},
    'reflection_backdoor_0.01_global_poisoning': {'type': 'reflection_backdoor', 'poison_ratio': 0.01, 'poison_ratio_wrt_class_members': False},
    'reflection_backdoor_0.03_global_poisoning': {'type': 'reflection_backdoor', 'poison_ratio': 0.03, 'poison_ratio_wrt_class_members': False},
    'reflection_backdoor_0.05_global_poisoning': {'type': 'reflection_backdoor', 'poison_ratio': 0.05, 'poison_ratio_wrt_class_members': False},
    'reflection_backdoor_0.1_global_poisoning': {'type': 'reflection_backdoor', 'poison_ratio': 0.1, 'poison_ratio_wrt_class_members': False},
    'reflection_backdoor_0.3_global_poisoning': {'type': 'reflection_backdoor', 'poison_ratio': 0.3, 'poison_ratio_wrt_class_members': False},
    'reflection_backdoor_0.5_global_poisoning': {'type': 'reflection_backdoor', 'poison_ratio': 0.5, 'poison_ratio_wrt_class_members': False},
    
    'clean_label_backdoor_global_poisoning': {'type': 'clean_label_backdoor', 'poison_ratio_wrt_class_members': False},
    'clean_label_backdoor_0_global_poisoning': {'type': 'clean_label_backdoor', 'poison_ratio': 0., 'poison_ratio_wrt_class_members': False},
    'clean_label_backdoor_0.01_global_poisoning': {'type': 'clean_label_backdoor', 'poison_ratio': 0.01, 'poison_ratio_wrt_class_members': False},
    'clean_label_backdoor_0.03_global_poisoning': {'type': 'clean_label_backdoor', 'poison_ratio': 0.03, 'poison_ratio_wrt_class_members': False},
    'clean_label_backdoor_0.05_global_poisoning': {'type': 'clean_label_backdoor', 'poison_ratio': 0.05, 'poison_ratio_wrt_class_members': False},
    'clean_label_backdoor_0.1_global_poisoning': {'type': 'clean_label_backdoor', 'poison_ratio': 0.1, 'poison_ratio_wrt_class_members': False},
    'clean_label_backdoor_0.3_global_poisoning': {'type': 'clean_label_backdoor', 'poison_ratio': 0.3, 'poison_ratio_wrt_class_members': False},
    'clean_label_backdoor_0.5_global_poisoning': {'type': 'clean_label_backdoor', 'poison_ratio': 0.5, 'poison_ratio_wrt_class_members': False},
    
    'wanet_backdoor_global_poisoning': {'type': 'wanet_backdoor', 'poison_ratio_wrt_class_members': False},
    'wanet_backdoor_0_global_poisoning': {'type': 'wanet_backdoor', 'poison_ratio': 0., 'poison_ratio_wrt_class_members': False},
    'wanet_backdoor_0.01_global_poisoning': {'type': 'wanet_backdoor', 'poison_ratio': 0.01, 'poison_ratio_wrt_class_members': False},
    'wanet_backdoor_0.03_global_poisoning': {'type': 'wanet_backdoor', 'poison_ratio': 0.03, 'poison_ratio_wrt_class_members': False},
    'wanet_backdoor_0.05_global_poisoning': {'type': 'wanet_backdoor', 'poison_ratio': 0.05, 'poison_ratio_wrt_class_members': False},
    'wanet_backdoor_0.1_global_poisoning': {'type': 'wanet_backdoor', 'poison_ratio': 0.1, 'poison_ratio_wrt_class_members': False},
    'wanet_backdoor_0.3_global_poisoning': {'type': 'wanet_backdoor', 'poison_ratio': 0.3, 'poison_ratio_wrt_class_members': False},
    'wanet_backdoor_0.5_global_poisoning': {'type': 'wanet_backdoor', 'poison_ratio': 0.5, 'poison_ratio_wrt_class_members': False},
    
    'horizontal_backdoor_global_poisoning': {'type': 'horizontal_backdoor', 'poison_ratio_wrt_class_members': False},
    'horizontal_backdoor_0_global_poisoning': {'type': 'horizontal_backdoor', 'poison_ratio': 0., 'poison_ratio_wrt_class_members': False},
    'horizontal_backdoor_0.01_global_poisoning': {'type': 'horizontal_backdoor', 'poison_ratio': 0.01, 'poison_ratio_wrt_class_members': False},
    'horizontal_backdoor_0.03_global_poisoning': {'type': 'horizontal_backdoor', 'poison_ratio': 0.03, 'poison_ratio_wrt_class_members': False},
    'horizontal_backdoor_0.05_global_poisoning': {'type': 'horizontal_backdoor', 'poison_ratio': 0.05, 'poison_ratio_wrt_class_members': False},
    'horizontal_backdoor_0.1_global_poisoning': {'type': 'horizontal_backdoor', 'poison_ratio': 0.1, 'poison_ratio_wrt_class_members': False},
    'horizontal_backdoor_0.3_global_poisoning': {'type': 'horizontal_backdoor', 'poison_ratio': 0.3, 'poison_ratio_wrt_class_members': False},
    'horizontal_backdoor_0.5_global_poisoning': {'type': 'horizontal_backdoor', 'poison_ratio': 0.5, 'poison_ratio_wrt_class_members': False},
    
}