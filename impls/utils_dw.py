import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from typing import Callable, Dict
import psutil
from torchvision import utils
from einops import rearrange, repeat
import imageio

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # Memory usage in MB

def get_available_ram():
    mem = psutil.virtual_memory()
    return mem.available / (1024 * 1024 * 1024)  # Available memory in MB

def dict_to_namespace(cfg_dict):
    args = argparse.Namespace()
    for key in cfg_dict:
        setattr(args, key, cfg_dict[key])
    return args

def move_to_device(dct, device):
    for key, value in dct.items():
        if isinstance(value, torch.Tensor):
            dct[key] = value.to(device)
    return dct

def slice_trajdict_with_t(data_dict, start_idx=0, end_idx=None, step=1):
    if end_idx is None:
        end_idx = max(arr.shape[1] for arr in data_dict.values())
    return {key: arr[:, start_idx:end_idx:step, ...] for key, arr in data_dict.items()}

def concat_trajdict(dcts):
    full_dct = {}
    for k in dcts[0].keys():
        if isinstance(dcts[0][k], np.ndarray):
            full_dct[k] = np.concatenate([dct[k] for dct in dcts], axis=1)
        elif isinstance(dcts[0][k], torch.Tensor):
            full_dct[k] = torch.cat([dct[k] for dct in dcts], dim=1)
        else:
            raise TypeError(f"Unsupported data type: {type(dcts[0][k])}")
    return full_dct

def aggregate_dct(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct

def sample_tensors(tensors, n, indices=None):
    if indices is None:
        b = tensors[0].shape[0]
        indices = torch.randperm(b)[:n]
    indices = torch.tensor(indices)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            tensors[i] = tensor[indices]
    return tensors


def cfg_to_dict(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    for key in cfg_dict:
        if isinstance(cfg_dict[key], list):
            cfg_dict[key] = ",".join(cfg_dict[key])
    return cfg_dict

def reduce_dict(f: Callable, d: Dict):
    return {k: reduce_dict(f, v) if isinstance(v, dict) else f(v) for k, v in d.items()}

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")
        

def save_video_from_rollout(e_visuals,goal_visual,successes,correction,task_id,save_path):
    for idx in range(e_visuals.shape[0]):
        success_tag = "success" if successes[idx] else "failure"
        frames = []
        for i in range(e_visuals.shape[1]):
            e_obs = e_visuals[idx, i, ...]
            e_obs = torch.cat(
                [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
            )
            frame = e_obs
            frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
            frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
            frame = frame.detach().cpu().numpy()
            frames.append(frame)
        video_writer = imageio.get_writer(
            f"{save_path}_{task_id}_{success_tag}.mp4", fps=12
        )

        for frame in frames:
            frame = frame * 2 - 1 if frame.min() >= 0 else frame
            video_writer.append_data(
                (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
            )
        video_writer.close()

    
    e_visuals = e_visuals[:, :: 5]


    n_columns = e_visuals.shape[1]

    e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
    rollout = e_visuals
 
    n_columns += 1

    imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
    imgs_for_plotting = (
        imgs_for_plotting * 2 - 1
        if imgs_for_plotting.min() >= 0
        else imgs_for_plotting
    )
    utils.save_image(
        imgs_for_plotting,
        f"{save_path}_{task_id}.png",
        nrow=n_columns, 
        normalize=True,
        value_range=(-1, 1),
    )
