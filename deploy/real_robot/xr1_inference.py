import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from lerobot.common.policies.xr1.modeling_xr1_stage2 import Xr1Stage2Policy
from lerobot.common.datasets.Humanoid_lerobot_dataset import HumanoidMultiRobotLeRobotDataset

from data.train_robot_script import section_dataset

device = "cuda"
xr1_stage2_pretrained_path = "../pretrained/XR-1-Stage2"
policy = Xr1Stage2Policy.from_pretrained(xr1_stage2_pretrained_path, map_location=device)
policy_config = policy.config
policy_config.n_obs_steps = 1
# our custom lerobot data
select_dataset = "XR_1_DATASET_SAMPLE_No_Ego4d"
ALL_TASK_CONFIG = section_dataset(select_dataset)

multi_dataset = HumanoidMultiRobotLeRobotDataset(ALL_TASK_CONFIG, policy_config)

dataloader = torch.utils.data.DataLoader(
    multi_dataset,
    num_workers=0,
    batch_size=1,
    shuffle=False,
    drop_last=False,
)

for batch in dataloader:
    observation = {
        "robot_type": batch['robot_type'],
        "task": batch['task']
    }
    for key in batch.keys():                            
        if key not in ["robot_type", "task"]:
            observation[key] = batch[key].to(device)

    with torch.inference_mode():
        pred_actions = policy.select_action(observation,action_horizon=50)
        print("pred_actions: ", pred_actions)





