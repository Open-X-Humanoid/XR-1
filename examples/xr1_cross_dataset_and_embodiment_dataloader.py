import torch
from lerobot.common.datasets.Humanoid_lerobot_dataset import HumanoidMultiRobotLeRobotDataset
from lerobot.common.policies.xr1.configuration_xr1_stage1 import Xr1Stage1Config

from data.train_robot_script import section_dataset
from lerobot.common.utils.utils import (
    format_big_number,
)

import time

# Dataset configuration
select_dataset = "XR_1_DATASET_SAMPLE"
ALL_TASK_CONFIG = section_dataset(select_dataset)

fake_config = Xr1Stage1Config()
multi_dataset = HumanoidMultiRobotLeRobotDataset(ALL_TASK_CONFIG, fake_config)

multi_dataset.num_frames = 0
multi_dataset.num_episodes = 0
for dataset in multi_dataset.multirobot_datasets:
    multi_dataset.num_frames += dataset['dataset'].num_frames
    multi_dataset.num_episodes += dataset['dataset'].num_episodes
print(f"{multi_dataset.num_frames=} ({format_big_number(multi_dataset.num_frames)})")
print(f"{multi_dataset.num_episodes=}")

batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader = torch.utils.data.DataLoader(
    multi_dataset,
    num_workers=0,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=device.type != "cpu",
    drop_last=False,
)

previous_time = time.time()
iteration = 0
for batch in dataloader:
    current_time = time.time()
    if iteration > 0:  
        iteration_time = current_time - previous_time
        print(f"Iteration {iteration} time: {iteration_time:.4f} seconds")

    camera_key = 'observation.images.image_0'
    print(f"{batch[camera_key].shape=}") 
    previous_time = time.time()

    iteration += 1


    
