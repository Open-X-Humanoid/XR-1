from lerobot.common.policies.xr1.modeling_xr1_stage2 import Xr1Stage2Policy
import numpy as np
import torch, cv2
import torch.nn.functional as F
class XR1_Evaluation(): 
    def __init__(self,model_path,robot_type,check_image_recons=False,real_robot_dev=False,dataset_stats_path=None,action_horizon=50,exp_weight=0.05,sampled_action_factor=1,ensemble=True):
        self.device = "cuda"
        self.policy = Xr1Stage2Policy.from_pretrained(model_path, map_location=self.device) 
            

        self.policy.config.pretrained_path = model_path
        self.policy.init_dataset_stats(self.device,robot_type,path=dataset_stats_path)
        self.action_execute_num = 0
        self.action_horizon = action_horizon
        self.policy.reset(action_horizon=action_horizon,
                          ensemble=ensemble,exp_weight=exp_weight,action_execute_steps=1,
                          sampled_action_factor=sampled_action_factor)
        self.policy.eval()

    def resize_with_pad(self, img, width, height, pad_value=-1):

        cur_height, cur_width = img.shape[-2:] #最后2位

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)  
            resized_img = F.interpolate(
                img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
            )
            resized_img = resized_img.squeeze(0)
        else:
            resized_img = F.interpolate(
                img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
            )

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
        return padded_img

    def Inference_Dual_Arm_Franka(self, obs, task_name):
        observation = dict()
        observation["task"] = [task_name]

        # images
        for cam_name in ['left','right','top','front']:
            cam_img = obs['images'][cam_name]
            cam_img = cv2.imdecode(cam_img, cv2.IMREAD_COLOR)
            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
            cam_img_resize = cv2.resize(cam_img, dsize=(640,480)) 
            cam_img_resize = torch.from_numpy(cam_img_resize).to(torch.float32)/255
            cam_img_resize = cam_img_resize.permute(2,0,1)
            
            cam_img_resize = cam_img_resize.unsqueeze(0).to(self.device)
            cam_img_resize = self.resize_with_pad(cam_img_resize, *self.policy.config.resize_imgs_with_padding, pad_value=0)
            
            if cam_name == 'front':
                observation["observation.images.image_0"] = cam_img_resize
            elif cam_name == 'left':
                observation["observation.images.image_1"] = cam_img_resize
            elif cam_name == 'right':
                observation["observation.images.image_2"] = cam_img_resize
            elif cam_name == 'top':
                observation["observation.images.image_3"] = cam_img_resize
            
        # state
        state_arm_left = torch.from_numpy(obs["arm_joints"]['left']).unsqueeze(0).to(self.device)
        state_arm_right = torch.from_numpy(obs["arm_joints"]['right']).unsqueeze(0).to(self.device)
        observation["observation.state.arm_joint_position"] = torch.cat([state_arm_left,state_arm_right],dim=1)

        action_queue = self.policy.select_action(observation,action_horizon=self.action_horizon)  
        return action_queue


def xr1_deploy():
    robot_type = "franka_dual" # deploy type: franka_dual
    model_name = "debug_output" # model output path
    task_name = "franka_dr3_stack_the_bowl" # task file name in sample dataset

    action_horizon=50 # action horizon
    exp_weight=0.05 # exp weight
    sampled_action_factor=1 # sampled action factor
    ensemble=False # ensemble
    pred_action_queue_num=0 # pred action queue number

    dataset_stats_path = f"./{model_name}/dataset_stats/{task_name}"
    model_path = f"./{model_name}/checkpoints/last/pretrained_model/"
    xr1_eval = XR1_Evaluation(model_path,robot_type,check_image_recons=False,real_robot_dev=False,dataset_stats_path=dataset_stats_path,
                                            action_horizon=action_horizon, exp_weight=exp_weight,sampled_action_factor=sampled_action_factor,ensemble=ensemble)


    # fake data
    episode_len = 500
    episode_qpos_arm_hand = np.random.rand(episode_len, 16).astype(np.float32)
    language_instruction = "stack the bowl"

    for index in range(episode_len):

        def get_fake_encoded_img():
            raw_img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
            _, encoded_img = cv2.imencode('.jpg', raw_img)
            return encoded_img

        fake_left_image = get_fake_encoded_img()
        fake_right_image = get_fake_encoded_img()
        fake_top_image = get_fake_encoded_img()
        fake_front_image = get_fake_encoded_img()
        fake_wrist_left_image = get_fake_encoded_img()
        fake_wrist_right_image = get_fake_encoded_img()

        fake_obs = {
            'images': {
                'left': fake_left_image,
                'right': fake_right_image,
                'top': fake_top_image,
                'front': fake_front_image,
                'wrist_left': fake_wrist_left_image,
                'wrist_right': fake_wrist_right_image
            },
            'arm_joints': {
                'left': episode_qpos_arm_hand[index][:7],
                'right': episode_qpos_arm_hand[index][7:]
            },
        }
        if pred_action_queue_num == 0:
            pred_action_queue = xr1_eval.Inference_Dual_Arm_Franka(fake_obs,language_instruction)
        pred_action = pred_action_queue.popleft()
        pred_action_queue_num = len(pred_action_queue)
        print("pred_action: ", pred_action)
if __name__ == "__main__":
    xr1_deploy()