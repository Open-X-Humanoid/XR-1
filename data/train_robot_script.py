
from pathlib import Path

found_repo_ids = []
task_config_dict = {}
def get_task_config(root_path: str, dataset_type: str, repo_ids: list | None=None, sample_weights = [1]):
    if repo_ids is None:
        repo_ids = [
            d.name for d in Path(root_path).iterdir() 
            if d.is_dir() and not d.name.startswith('.') # ignore hidden files
        ]
    if dataset_type == "oxe":
        if sample_weights == [1]:
            sample_weights = [10]
        else:
            sample_weights = sample_weights
    task_config = {
        'multi_task': {
            'repo_ids': repo_ids,
            'root': root_path,
            'dataset_type': dataset_type
        },
        'sample_weights': sample_weights * len(repo_ids)
    }
    return task_config
    

def select_xr1_dataset_sample():
    root = "../dataset/XR-1-Dataset-Sample"
    '''DUAL_ARM_FRANKA'''
    task_config_dual_arm_franka = get_task_config(f"{root}/DUAL_ARM_FRANKA", "humanoid_station")
    '''DUAL_ARM_UR'''
    task_config_dual_arm_ur = get_task_config(f"{root}/DUAL_ARM_UR", "humanoid_station")
    '''SINGLE_ARM_UR'''
    task_config_single_arm_ur = get_task_config(f"{root}/SINGLE_ARM_UR", "humanoid_station")
    '''DUAL_ARM_AGILEX'''   
    task_config_dual_arm_agx = get_task_config(f"{root}/DUAL_ARM_AGX", "humanoid_station")
    '''DUAL_ARM_TIEN_KUNG2'''
    task_config_dual_arm_tien_kung2 = get_task_config(f"{root}/DUAL_ARM_TIEN_KUNG2", "humanoid_station")
    '''EGO4D'''
    task_config_ego4d = {
        'multi_task': {
            'repo_ids': "action_clips_new_info.json",
            'root': f"{root}/EGO4D/",
            'dataset_type': 'ego4d'
        },
        'sample_weights': [1] * 10
    }  

    TASK_CONFIGS = [
            # '''dual_arm_franka'''
            task_config_dual_arm_franka,
            # '''dual_arm_ur'''
            task_config_dual_arm_ur,
            # '''single_arm_ur'''
            task_config_single_arm_ur,
            # '''dual_arm_agilex'''
            task_config_dual_arm_agx,
            # '''dual_arm_tien_kung2'''
            task_config_dual_arm_tien_kung2,
            # '''ego4d'''
            task_config_ego4d,
    ]

    ALL_TASK_CONFIG = {
        'multi_task': [cfg['multi_task'] for cfg in TASK_CONFIGS],
        'sample_weights': [cfg['sample_weights'] for cfg in TASK_CONFIGS],
    }
    return ALL_TASK_CONFIG

def select_xr1_dataset_sample_no_ego4d():
    root = "../dataset/XR-1-Dataset-Sample"
    '''DUAL_ARM_FRANKA'''
    task_config_dual_arm_franka = get_task_config(f"{root}/DUAL_ARM_FRANKA", "humanoid_station")
    '''DUAL_ARM_UR'''
    task_config_dual_arm_ur = get_task_config(f"{root}/DUAL_ARM_UR", "humanoid_station")
    '''SINGLE_ARM_UR'''
    task_config_single_arm_ur = get_task_config(f"{root}/SINGLE_ARM_UR", "humanoid_station")
    '''DUAL_ARM_AGILEX'''
    task_config_dual_arm_agx = get_task_config(f"{root}/DUAL_ARM_AGX", "humanoid_station")
    '''DUAL_ARM_TIEN_KUNG2'''
    task_config_dual_arm_tien_kung2 = get_task_config(f"{root}/DUAL_ARM_TIEN_KUNG2", "humanoid_station")


    TASK_CONFIGS = [
            # '''dual_arm_franka'''
            task_config_dual_arm_franka,
            # '''dual_arm_ur'''
            task_config_dual_arm_ur,
            # '''single_arm_ur'''
            task_config_single_arm_ur,
            # '''dual_arm_agilex'''
            task_config_dual_arm_agx,
            # '''dual_arm_tien_kung2'''
            task_config_dual_arm_tien_kung2,
    ]

    ALL_TASK_CONFIG = {
        'multi_task': [cfg['multi_task'] for cfg in TASK_CONFIGS],
        'sample_weights': [cfg['sample_weights'] for cfg in TASK_CONFIGS],
    }
    return ALL_TASK_CONFIG

def select_xr1_dataset_Dual_Arm_Franka():
    root = "../dataset/XR-1-Dataset-Sample"
    '''DUAL_ARM_FRANKA'''
    task_config_dual_arm_franka = get_task_config(f"{root}/DUAL_ARM_FRANKA", "humanoid_station")

    TASK_CONFIGS = [
            # '''dual_arm_franka'''
            task_config_dual_arm_franka,
    ]

    ALL_TASK_CONFIG = {
        'multi_task': [cfg['multi_task'] for cfg in TASK_CONFIGS],
        'sample_weights': [cfg['sample_weights'] for cfg in TASK_CONFIGS],
    }
    return ALL_TASK_CONFIG

def section_dataset(select_dataset: str, repo_ids=None):

    if select_dataset == "XR_1_DATASET_SAMPLE":
        ALL_TASK_CONFIG = select_xr1_dataset_sample()
        return ALL_TASK_CONFIG
    elif select_dataset == "XR_1_DATASET_SAMPLE_No_Ego4d":
        ALL_TASK_CONFIG = select_xr1_dataset_sample_no_ego4d()
        return ALL_TASK_CONFIG
    elif select_dataset == "XR_1_DATASET_DUAL_ARM_FRANKA":
        ALL_TASK_CONFIG = select_xr1_dataset_Dual_Arm_Franka()
        return ALL_TASK_CONFIG
    
    raise ValueError(f"Invalid dataset selection: {select_dataset}")
    
