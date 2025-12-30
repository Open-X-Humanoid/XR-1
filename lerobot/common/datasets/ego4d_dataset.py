import os
import json
import random
import logging
from pathlib import Path
from typing import List, Tuple, Union
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torchvision.io import VideoReader

def decode_video_frames_torchvision(
    video_path: Union[Path, str],
    timestamps: List[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """
    Loads the frames closest to the requested timestamps using torchvision.io.VideoReader
    with the 'pyav' backend. Returns a tensor of shape (len(timestamps), C, H, W),
    dtype=float32 in [0,1].
    """
    # force pyav backend
    torchvision.set_video_backend("pyav")
    keyframes_only = True

    reader = VideoReader(str(video_path), "video")
    first_ts, last_ts = min(timestamps), max(timestamps)
    reader.seek(first_ts, keyframes_only=keyframes_only)

    loaded_frames: List[torch.Tensor] = []
    loaded_ts: List[float] = []
    for frame in reader:
        ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"Loaded frame at ts={ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(ts)
        if ts >= last_ts:
            break

    reader.container.close()

    query_ts = torch.tensor(timestamps)
    loaded_ts_tensor = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
    min_dist, argmin = dist.min(1)
    assert (min_dist < tolerance_s).all(), (
        f"Requested timestamps too far from decoded frames: {min_dist}"
    )

    closest = torch.stack([loaded_frames[i] for i in argmin])
    return closest.to(torch.float32).div(255.0)


class Ego4DDataset(Dataset):
    def __init__(
        self,
        json_path: str = "mp4_files_info.json",
        target_size: Tuple[int, int] = (640, 480),
        interval: int = 50,
        split_dataset: bool = False,
        downsample_factor: int = 1,
    ):
        """
        Args:
            json_path: JSON file with entries [{"path":..., "num_frames":..., "fps":...}, ...]
            target_size: (H, W) to resize each frame
            interval: frame-index gap between the two sampled frames
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not found")
        with open(json_path, "r", encoding="utf-8") as f:
            self.video_list_total = json.load(f)
        if not isinstance(self.video_list_total, list) or not self.video_list_total:
            raise ValueError(f"{json_path} contains no valid entries")
        if split_dataset:
            rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 80))
            self.video_list = self.video_list_total[rank::world_size]
            self.video_list = self.video_list[::downsample_factor]
        else:
            self.video_list = self.video_list_total
        self.interval   = interval
        self.tolerance_s = 1.0 / 30.0  # ~33ms tolerance
        self.resize     = transforms.Resize(target_size)
        self.to_tensor  = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.video_list)

    def __getitem__(self, idx=None) -> torch.Tensor:
        entry = random.choice(self.video_list)
        path = entry["path"]
        num_frames = entry.get("num_frames")
        fps = float(entry.get("fps"))

        if num_frames < self.interval + 1:
            raise RuntimeError(
                f"{path} has only {num_frames} frames, need ≥{self.interval+1}"
            )
        
        # 初始帧随机选择
        first_idx = random.randint(0, num_frames - self.interval - 1)
        second_idx = first_idx + self.interval
        t1, t2 = first_idx / fps, second_idx / fps

        frames = decode_video_frames_torchvision(
            video_path=path,
            timestamps=[t1, t2],
            tolerance_s=self.tolerance_s,
        )  # (2, C, H0, W0) float32 in [0,1]

        processed = []
        for f in frames:
            pil = transforms.functional.to_pil_image(f)
            # processed.append(self.to_tensor(self.resize(pil))) # resize 到 (640, 480)
            processed.append(self.to_tensor(pil))
        item = {}
        image_0 = torch.stack(processed, dim=0)
        item['observation.images.image_0'] = image_0
        item['observation.images.image_0_is_pad'] = torch.zeros(image_0.shape[0], dtype=torch.bool)

        # result: dict(2, C, H, W)
        return item
    @property
    def num_episodes(self) -> int:
        return len(self.video_list)
    
    @property   
    def num_frames(self) -> int:
        """Number of samples/frames."""
        # sum所有视频的帧数
        return sum(entry.get("num_frames") for entry in self.video_list)
    
def save_batch_frames(batch: torch.Tensor, out_dir: str = 'tmp'):
    """
    将一个 batch 内的所有帧拆解并保存为图片。
    
    Args:
        batch: Tensor of shape (B, 2, C, H, W),  float in [0,1] or [0,255]
        out_dir: 保存目录，会自动创建
    """
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
    
    B, F, C, H, W = batch.shape
    # 如果 tensor 是 float 且 [0,1]，先转到 [0,255] uint8
    if batch.dtype in (torch.float32, torch.float64):
        batch_uint8 = (batch.clamp(0,1) * 255).byte()
    else:
        batch_uint8 = batch.byte()
    
    for b in range(B):
        for f in range(F):
            img_tensor = batch_uint8[b, f]               # shape (C, H, W)
            # 转成 H×W×C 的 NumPy 数组
            img_arr = img_tensor.permute(1, 2, 0).cpu().numpy()
            # PIL Image
            img = Image.fromarray(img_arr)
            # 构造文件名：比如 tmp/batch0_frame1.png
            fn = os.path.join(out_dir, f"batch{b}_frame{f}.png")
            img.save(fn)
    print(f"Saved {B*F} images to '{out_dir}/'")


if __name__ == "__main__":
    # example usage
    from torch.utils.data import DataLoader

    ds = Ego4DDataset(
        json_path="examples/mp4_files_info.json",
        target_size=(640, 480),
        interval=50,
    )
    sample = ds[0]
    print("Sample shape:", sample.shape)  # e.g. torch.Size([2, 3, 480, 640])

    loader = DataLoader(ds, batch_size=12, shuffle=True, num_workers=0)
    
    for batch in loader:
        # batch.shape == (2, 2, 3, 480, 640)
        print("Batch shape:", batch.shape)
        # 保存到 tmp/
        # save_batch_frames(batch, out_dir='tmp')
        break
