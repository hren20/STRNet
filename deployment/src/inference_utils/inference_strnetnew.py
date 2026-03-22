# inference.py
import os
import torch
import yaml
import argparse
import numpy as np
from typing import Dict, List, Tuple, Union
from torchvision import transforms
from PIL import Image as PILImage
from diffusers import DDPMScheduler

# 复用训练框架组件
from vint_train.training.trainer import *
from vint_train.training.logger import Logger
from vint_train.training.trainmanager import *

from utils_inference import transform_images
from .inference_base import BaseInferenceTrainer
from .common import to_numpy

class InferenceSTRNetNewTrainer(STRNetNewTrainer, BaseInferenceTrainer):
    """扩展训练器支持推理功能"""
    
    def __init__(self, config: Dict, checkpoint_path: str):
        super().__init__(config)
        self._load_inference_model(checkpoint_path)
        self.pred_horizon = config["len_traj_pred"]
        self.action_dim = 2
        self.init_inference_action_stats()

    def _load_inference_model(self, checkpoint_path: str):
        """加载推理模型（复用训练器方法）"""
        load_model(
            model=self.model,
            model_type=self.config["model_type"],
            checkpoint=torch.load(checkpoint_path, map_location=self.device)
        )
        self.model = self.model.to(self.device).eval()
        self.ema_model = EMAModel(self.model, power=0.75)
        print(f"Loaded model from {checkpoint_path}")

    def prepare_inputs(self, image_context: List[Union[str, PILImage.Image]]) -> torch.Tensor:
        """图像路径或PIL图像列表 -> 模型输入张量 (B, C, H, W)"""
        images = []
        for img in image_context:
            if isinstance(img, str):
                img = PILImage.open(img).convert("RGB")
            else:
                img = img.convert("RGB")
            images.append(img)

        obs_tensor = transform_images(images, self.config["image_size"], center_crop=False)
        return obs_tensor.to(self.device)

    @torch.no_grad()
    def predict_actions(self, obs_images: torch.Tensor, goal_images: torch.Tensor = None, num_samples: int = 100) -> Dict:
        """完整推理流程（复用训练生成方法）"""
        # 生成空白目标图像
        if goal_images is None:
            goal_images = torch.randn((1, 3, *self.config["image_size"])).to(self.device)

        # 复用训练器的扩散生成方法
        results = self._denoise_loop(
            model=self.ema_model.averaged_model,
            obs=obs_images,
            goal=goal_images,  # 全目标条件
            num_samples=num_samples
        )

        actions, dists = results["actions"], results["dists"]

        # 后处理
        actions = to_numpy(self._get_action(actions, self.ACTION_STATS))
        dists = to_numpy(dists.flatten())
        return actions, dists

    @torch.no_grad()
    def goal_denosing(self, model, cond, num_samples):
        if len(cond.shape) == 2:
            cond = cond.repeat(num_samples, 1)
        else:
            cond = cond.repeat(num_samples, 1, 1)
        # 初始化噪声动作
        noisy_actions = torch.randn(
            (cond.size(0), self.pred_horizon, self.action_dim), 
            device=self.device
        )

        # 迭代去噪
        for t in self.noise_scheduler.timesteps:
            noise_pred = model.noise_pred_net(
                sample=noisy_actions,
                timestep=torch.full((noisy_actions.size(0),), t, device=self.device),
                global_cond=cond
            )
            noisy_actions = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_actions
            ).prev_sample

        return noisy_actions

    @torch.no_grad()
    def topo_chosen(self, model, obs, goal, num_samples) -> torch.Tensor:
        cond = model.vision_encoder(obs, goal)
        dists = model("dist_pred_net", obsgoal_cond=cond)
        dists = to_numpy(dists.flatten())
        min_idx = np.argmin(dists)
        sg_idx = min(min_idx + int(dists[min_idx] < self.config["close_threshold"]), len(cond) - 1)
        obs_cond = cond[sg_idx].unsqueeze(0)
        return obs_cond, min_idx

    @torch.no_grad()
    def toponavi_policy(self, obs_images: torch.Tensor, goal_images: torch.Tensor = None, num_samples: int = 100) -> Dict:
        """
        推理接口，仅返回 goal-conditioned actions 和 predicted distance
        """
        model = self.ema_model.averaged_model

        # 如果未指定 goal image，则使用 obs 最后3通道
        if goal_images is None:
            goal_images = obs_images[:, -3:, :, :]

        batch_size = obs_images.size(0)

        obs_cond, min_idx = self.topo_chosen(model, obs=obs_images, goal=goal_images, num_samples=num_samples)
        actions = self.goal_denosing(model=model, cond=obs_cond, num_samples=num_samples)

        # 后处理
        actions = to_numpy(self._get_action(actions, self.ACTION_STATS))

        return actions, min_idx


    def _postprocess(self, actions: torch.Tensor) -> np.ndarray:
        """复用训练器的动作后处理"""
        return self._get_action(actions, self.ACTION_STATS).squeeze(0).cpu().numpy()