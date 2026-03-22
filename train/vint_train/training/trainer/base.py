import os
import torch
import wandb
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import itertools
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
    CyclicLR,
    ReduceLROnPlateau,
    StepLR
)
from warmup_scheduler import GradualWarmupScheduler

from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm  # 自动适配Jupyter/终端环境
import yaml
import numpy as np
import matplotlib.pyplot as plt

from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from vint_train.training.logger import Logger

from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy

from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.navibridge.navibridge import NaviBridge, DenseNetwork, StatesPredNet
from vint_train.models.navibridge.navibridg_utils import NaviBridge_Encoder, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from vint_train.models.navibridge.ddbm.karras_diffusion import KarrasDenoiser
from vint_train.models.navibridge.ddbm.resample import create_named_schedule_sampler
from vint_train.models.navibridge.navibridge import PriorModel, Prior_HandCraft
from vint_train.models.navibridge.vae.vae import VAEModel

class BaseTrainer(ABC):
    """训练器基类，封装通用训练逻辑"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config["device"]    # 由策略类注入
        if self.config["train"]:
            self.project_folder = config["project_folder"]
            self.eval_fraction = config.get("eval_fraction", 1.0)
        self.transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(self.transform)
        self.ema_model = None
        self._init_components()
        self.loggers = self._create_loggers()
        self._init_logging_config()
        
    def _init_components(self):
        """初始化模型、优化器等组件"""
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)
        self.loggers = self._create_loggers()

    def _init_logging_config(self):
        """初始化日志相关配置"""
        self.logging_config = {
            'print_freq': self.config.get("print_log_freq", 50),
            'wandb_freq': self.config.get("wandb_log_freq", 10),
            'image_freq': self.config.get("image_log_freq", 100),
            'num_images': self.config.get("num_images_log", 8),
            'use_wandb': self.config.get("use_wandb", False)
        }

    def _log_batch_metrics(self, batch_idx: int, total_batches: int, epoch: int, mode: str):
        """统一批处理日志"""
        # 控制台日志
        if batch_idx % self.logging_config['print_freq'] == 0:
            self._print_batch_metrics(epoch, batch_idx, total_batches, mode)

        # WandB日志
        if self.logging_config['use_wandb'] and batch_idx % self.logging_config['wandb_freq'] == 0:
            self._log_wandb_metrics(mode)

    def _print_batch_metrics(self, epoch: int, batch_idx: int, total_batches: int, mode: str):
        """打印控制台日志"""
        log_str = f"[{mode.upper()}] Epoch {epoch} Batch {batch_idx}/{total_batches}\n"
        for name, logger in self.loggers.items():
            if logger.dataset == mode:
                log_str += f"{logger.display()}\n"
        print(log_str)

    def _log_wandb_metrics(self, mode: str):
        """记录WandB指标"""
        metrics = {
            f"{mode}/{k}": v.avg 
            for k, v in self.loggers.items()
            if v.dataset == mode
        }
        wandb.log(metrics)

    def _update_progress_bar(self, mode: str):
        """更新进度条显示"""
        if mode == "train":
            loss_value = self.loggers["total_loss"].avg
            lr = self.optimizer.param_groups[0]['lr']
            self.progress_bar.set_postfix({"loss": f"{loss_value:.4f}", "lr": f"{lr:.2e}"})
        else:
            loss_value = self.loggers[f"total_loss"].avg
            self.progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

    def _validate_config(self, config: Dict, required_keys: List[str]):
        """验证配置完整性"""
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(
                f"配置缺少必需参数: {missing_keys}\n"
                f"当前配置包含的键: {list(config.keys())}"
            )

    def _load_action_stats(self):
        # LOAD DATA CONFIG
        with open(os.path.join(os.path.dirname(__file__), "../../data/data_config.yaml"), "r") as f:
            data_config = yaml.safe_load(f)
        # POPULATE ACTION STATS
        self.ACTION_STATS = {}
        for key in data_config['action_stats']:
            self.ACTION_STATS[key] = np.array(data_config['action_stats'][key])

    @abstractmethod
    def _create_model(self) -> torch.nn.Module:
        """创建模型实例"""
        pass
    
    @abstractmethod
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        pass

    @abstractmethod
    def _create_scheduler(self) -> LRScheduler:
        """创建优化器"""
        pass

    @abstractmethod
    def _create_loggers(self) -> Dict[str, Any]:
        """创建日志记录器"""
        pass
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """标准训练流程"""
        self.model.train()
        total_batches = len(dataloader)
        
        with self._iter_batches(dataloader, f"Training Epoch {epoch}") as progress_bar:
            self.progress_bar = progress_bar
            for batch_idx, batch in enumerate(progress_bar):
                losses = self._train_step(batch, epoch, batch_idx)
                self._update_loggers(losses)
                self._log_batch_metrics(batch_idx, total_batches, epoch, "train")
                self._update_progress_bar("train")

    # def evaluate(self, dataloader: DataLoader, epoch: int, eval_type: str) -> Dict[str, float]:
    #     """标准评估流程"""
    #     self.model.eval()
    #     total_batches = len(dataloader)
        
    #     with self._iter_batches(dataloader, f"Evaluating {eval_type}") as progress_bar, \
    #         torch.no_grad():
    #             self.progress_bar = progress_bar
    #             for batch_idx, batch in enumerate(progress_bar):
    #                 losses = self._eval_step(batch, epoch, batch_idx, eval_type)
    #                 self._update_eval_loggers(losses, eval_type)
    #                 self._log_batch_metrics(batch_idx, total_batches, epoch, eval_type)
    #                 self._update_progress_bar(eval_type)

    #     return {
    #         metric: logger.avg 
    #         for metric, logger in self.loggers.items()
    #         if logger.dataset == eval_type
    #     }

    def evaluate(self, dataloader: DataLoader, epoch: int, eval_type: str) -> Dict[str, float]:
        """标准评估流程"""
        self.model.eval()
        total_batches = len(dataloader)
        num_batches = max(int(total_batches * self.eval_fraction), 1)  # ✨ 计算实际评估批次
        
        # ✨ 创建有限批次的数据加载器
        limited_dataloader = itertools.islice(dataloader, num_batches)
        
        with self._iter_batches(limited_dataloader, 
                              f"Evaluating {eval_type}", 
                              total=num_batches) as progress_bar, \
            torch.no_grad():
                self.progress_bar = progress_bar
                for batch_idx, batch in enumerate(progress_bar):
                    losses = self._eval_step(batch, epoch, batch_idx, eval_type)
                    self._update_eval_loggers(losses, eval_type)
                    # ✨ 使用实际评估批次总数替代原始总数
                    self._log_batch_metrics(batch_idx, num_batches, epoch, eval_type)  
                    self._update_progress_bar(eval_type)

        return {
            metric: logger.avg 
            for metric, logger in self.loggers.items()
            if logger.dataset == eval_type
        }

    def _iter_batches(self, dataloader: DataLoader, desc: str, total: int = None):  # ✨ 新增total参数
        """带进度条的批次迭代器"""
        return tqdm(dataloader, desc=desc, total=total, dynamic_ncols=True)  # ✨ 传递total参数

    def _normalize_data(self, data, stats):
        # nomalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

    def _unnormalize_data(self, ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data

    def _get_delta(self, actions):
        # append zeros to first action
        ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
        delta = ex_actions[:,1:] - ex_actions[:,:-1]
        return delta

    def _get_action(self, diffusion_output, action_stats):
        # diffusion_output: (B, 2*T+1, 1)
        # return: (B, T-1)
        device = diffusion_output.device
        ndeltas = diffusion_output
        ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
        ndeltas = to_numpy(ndeltas)
        ndeltas = self._unnormalize_data(ndeltas, action_stats)
        actions = np.cumsum(ndeltas, axis=1)
        return from_numpy(actions).to(device)

    def _should_visualize(self, batch_idx: int, freq: int) -> bool:
        """判断是否需要可视化"""
        return (batch_idx % freq == 0) and (self.config["num_images_log"] > 0)

    @abstractmethod
    def _update_loggers(self, losses: dict):
        pass

    @abstractmethod
    def _update_eval_loggers(self, losses: dict, eval_type: str):
        pass

    @abstractmethod
    def _train_step(self, batch: Any, epoch: int, batch_idx: int):
        """单个训练步骤的具体实现"""
        pass
    
    @abstractmethod
    def _eval_step(self, batch: any, epoch: int, batch_idx: int, eval_type: str):
        """单个评估步骤的具体实现"""
        pass
    
    # def _iter_batches(self, dataloader: DataLoader, desc: str):
    #     """带进度条的批次迭代器"""
    #     return tqdm(dataloader, desc=desc, dynamic_ncols=True)
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """统一日志记录"""
        if self.config["use_wandb"]:
            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)
            
    def _visualize_results(self, batch: Any, epoch: int, mode: str):
        """统一可视化接口"""
        pass

# class ViNTTrainer(BaseTrainer):
#     """ViNT/GNM 训练器实现"""
    
#     def _create_model(self) -> torch.nn.Module:
#         """根据配置创建ViNT模型实例"""
#         model_config = self.config
        
#         # 验证必需参数
#         required_params = [
#             "context_size", 
#             "len_traj_pred",
#             "learn_angle",
#             "obs_encoder",
#             "obs_encoding_size",
#             "late_fusion",
#             "mha_num_attention_heads",
#             "mha_num_attention_layers",
#             "mha_ff_dim_factor",
#         ]
#         self._validate_config(model_config, required_params)
        
#         # 构建ViNT模型
#         return ViNT(
#             context_size=model_config["context_size"],
#             len_traj_pred=model_config["len_traj_pred"],
#             learn_angle=model_config["learn_angle"],
#             obs_encoder=model_config["obs_encoder"],
#             obs_encoding_size=model_config["obs_encoding_size"],
#             late_fusion=model_config["late_fusion"],
#             mha_num_attention_heads=model_config["mha_num_attention_heads"],
#             mha_num_attention_layers=model_config["mha_num_attention_layers"],
#             mha_ff_dim_factor=model_config["mha_ff_dim_factor"]
#         ).to(self.device)

#     def _create_optimizer(self) -> torch.optim.Optimizer:
#         """根据配置创建优化器"""
#         optimizer_type = self.config.get("optimizer", "adam").lower()
#         lr = float(self.config.get("lr", 1e-4))
#         weight_decay = self.config.get("weight_decay", 0)

#         param_groups = [
#             {'params': self.model.parameters()},
#             # 可扩展参数分组逻辑
#         ]

#         if optimizer_type == "adam":
#             return Adam(
#                 param_groups,
#                 lr=lr,
#                 betas=tuple(self.config.get("adam_betas", (0.9, 0.98))),
#                 weight_decay=weight_decay
#             )
#         elif optimizer_type == "adamw":
#             return AdamW(param_groups, lr=lr, weight_decay=weight_decay)
#         elif optimizer_type == "sgd":
#             return SGD(
#                 param_groups,
#                 lr=lr,
#                 momentum=self.config.get("momentum", 0.9),
#                 weight_decay=weight_decay
#             )
#         raise ValueError(f"Unsupported optimizer: {optimizer_type}")

#     def _create_scheduler(self, optimizer: torch.optim.Optimizer):
#         """根据配置创建学习率调度器"""
#         scheduler_type = self.config.get("scheduler", "").lower()
#         if not scheduler_type:
#             return None

#         # 基础调度器
#         if scheduler_type == "cosine":
#             return CosineAnnealingLR(
#                 optimizer,
#                 T_max=self.config.get("epochs", 100)
#             )
#         elif scheduler_type == "cyclic":
#             return CyclicLR(
#                 optimizer,
#                 base_lr=self.config["lr"] / 10.0,
#                 max_lr=self.config["lr"],
#                 step_size_up=self.config.get("cyclic_period", 50) // 2,
#                 cycle_momentum=False
#             )
#         elif scheduler_type == "plateau":
#             return ReduceLROnPlateau(
#                 optimizer,
#                 factor=self.config.get("plateau_factor", 0.1),
#                 patience=self.config.get("plateau_patience", 10),
#                 verbose=True
#             )
        
#         # Warmup包装
#         if self.config.get("warmup", False):
#             return GradualWarmupScheduler(
#                 optimizer,
#                 multiplier=1,
#                 total_epoch=self.config.get("warmup_epochs", 5),
#                 after_scheduler=self._create_base_scheduler(optimizer)
#             )
#         return None

#     def _create_loggers(self) -> Dict[str, Logger]:
#         loggers = {
#             "dist_loss": Logger("dist_loss", "train"),
#             "action_loss": Logger("action_loss", "train"),
#             "total_loss": Logger("total_loss", "train"),
#             "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "train"),
#             "multi_action_waypts_cos_sim": Logger("multi_action_waypts_cos_sim", "train")
#         }
        
#         if self.config["learn_angle"]:
#             loggers.update({
#                 "action_orien_cos_sim": Logger("action_orien_cos_sim", "train"),
#                 "multi_action_orien_cos_sim": Logger("multi_action_orien_cos_sim", "train")
#             })
#         return loggers
    
#     def _preprocess_batch(self, batch: tuple) -> tuple:
#         """数据预处理"""
#         (
#             obs_image, 
#             goal_image, 
#             action_label, 
#             dist_label, 
#             goal_pos, 
#             dataset_index, 
#             action_mask,
#             _
#         ) = batch
        
#         # 处理观测图像
#         obs_images = torch.split(obs_image, 3, dim=1)
#         viz_obs = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
#         obs_images = [self.transform(img).to(self.device) for img in obs_images]
#         obs_tensor = torch.cat(obs_images, dim=1)
        
#         # 处理目标图像
#         viz_goal = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
#         goal_tensor = self.transform(goal_image).to(self.device)
        
#         return (
#             obs_tensor, 
#             goal_tensor,
#             action_label.to(self.device),
#             dist_label.to(self.device),
#             action_mask.to(self.device),
#             (viz_obs, viz_goal, goal_pos, dist_label, action_label, dataset_index)
#         )
    
#     def _compute_losses(
#         self,  # 添加self参数
#         dist_label: torch.Tensor,
#         action_label: torch.Tensor,
#         dist_pred: torch.Tensor,
#         action_pred: torch.Tensor,
#         alpha: float,
#         learn_angle: bool,
#         action_mask: torch.Tensor = None,
#     ) -> Dict[str, torch.Tensor]:
#         # 使用配置参数代替直接传参
#         alpha = self.config["alpha"]
#         learn_angle = self.config["learn_angle"]
#         dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

#         def action_reduce(unreduced_loss: torch.Tensor):
#             # Reduce over non-batch dimensions to get loss per batch element
#             while unreduced_loss.dim() > 1:
#                 unreduced_loss = unreduced_loss.mean(dim=-1)
#             assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
#             return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

#         # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
#         assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
#         action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

#         action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
#             action_pred[:, :, :2], action_label[:, :, :2], dim=-1
#         ))
#         multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
#             torch.flatten(action_pred[:, :, :2], start_dim=1),
#             torch.flatten(action_label[:, :, :2], start_dim=1),
#             dim=-1,
#         ))

#         results = {
#             "dist_loss": dist_loss,
#             "action_loss": action_loss,
#             "action_waypts_cos_sim": action_waypts_cos_similairity,
#             "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
#         }

#         if learn_angle:
#             action_orien_cos_sim = action_reduce(F.cosine_similarity(
#                 action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
#             ))
#             multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
#                 torch.flatten(action_pred[:, :, 2:], start_dim=1),
#                 torch.flatten(action_label[:, :, 2:], start_dim=1),
#                 dim=-1,
#                 )
#             )
#             results["action_orien_cos_sim"] = action_orien_cos_sim
#             results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

#         total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
#         results["total_loss"] = total_loss

#         return results

#     def _train_step(self, batch: tuple, epoch: int, batch_idx: int):
#         """单训练步骤"""
#         # 数据预处理
#         obs_tensor, goal_tensor, action_label, dist_label, action_mask, viz_data = self._preprocess_batch(batch)
        
#         # 前向传播
#         outputs = self.model(obs_tensor, goal_tensor)
        
#         dist_pred, action_pred = outputs  # 假设模型返回元组
        
#         losses = self._compute_losses(
#             dist_label=dist_label,
#             action_label=action_label,
#             dist_pred=dist_pred,
#             action_pred=action_pred,
#             alpha = self.config["alpha"],
#             learn_angle=self.config["learn_angle"],
#             action_mask=action_mask
#         )
        
#         # 反向传播
#         self.optimizer.zero_grad()
#         losses["total_loss"].backward()
#         self.optimizer.step()
        
#         # 更新日志
#         self._update_loggers(losses)
        
#         # 可视化
#         if self._should_visualize(batch_idx, self.config["image_log_freq"]):
#             self._visualize_step(viz_data, outputs, epoch, "train")
#             # self._visualize_step(viz_data, outputs, losses, epoch, "train")
#         return losses

#     def _eval_step(self, batch: tuple, epoch, batch_idx: int, eval_type: str):
#         """单评估步骤"""
#         # 数据预处理
#         obs_tensor, goal_tensor, action_label, dist_label, action_mask, viz_data = self._preprocess_batch(batch)
        
#         # 前向传播
#         outputs = self.model(obs_tensor, goal_tensor)
        
#         dist_pred, action_pred = outputs 

#         losses = self._compute_losses(
#             dist_label=dist_label,
#             action_label=action_label,
#             dist_pred=dist_pred,
#             action_pred=action_pred,
#             alpha = self.config["alpha"],
#             learn_angle=self.config["learn_angle"],
#             action_mask=action_mask,
#         )

#         # 更新评估日志
#         self._update_eval_loggers(losses, eval_type)
        
#         # 可视化
#         if self._should_visualize(batch_idx, self.config["image_log_freq"]):
#             self._visualize_step(viz_data, outputs, epoch, eval_type)
#         return losses

#     def _update_loggers(self, losses: dict):
#         """统一使用log_data方法"""
#         for key in self.loggers:
#             if key in losses:
#                 # 修改此处调用方式
#                 self.loggers[key].log_data(losses[key].item())
    
#     def _update_eval_loggers(self, losses: dict, eval_type: str):
#         """修正方法名调用"""
#         for key in losses:
#             metric_name = f"{eval_type}/{key}"
#             if metric_name not in self.loggers:
#                 self.loggers[metric_name] = Logger(key, eval_type)
#             # 将update改为log_data
#             self.loggers[metric_name].log_data(losses[key].item())
    
#     def _visualize_step(self, viz_data: tuple, outputs: tuple, epoch: int, mode: str):
#         """统一可视化日志"""
#         viz_obs, viz_goal, goal_pos, dist_label, action_label, dataset_index = viz_data
#         dist_pred, action_pred = outputs

#         # 距离预测可视化
#         visualize_dist_pred(
#             to_numpy(viz_obs),
#             to_numpy(viz_goal),
#             to_numpy(dist_pred),
#             to_numpy(dist_label), 
#             mode,
#             self.project_folder,
#             epoch,
#             self.logging_config['num_images'],
#             use_wandb=self.logging_config['use_wandb']
#         )

#         # 轨迹预测可视化
#         visualize_traj_pred(
#             to_numpy(viz_obs),
#             to_numpy(viz_goal),
#             to_numpy(dataset_index),
#             to_numpy(goal_pos),
#             to_numpy(action_pred),
#             to_numpy(action_label),
#             mode,
#             self.config.get("normalized", False),
#             self.project_folder,
#             epoch,
#             self.logging_config['num_images'],
#             use_wandb=self.logging_config['use_wandb']
#         )

# class NoMaDTrainer(BaseTrainer):
#     """NoMaD扩散模型训练器实现"""
    
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         self.ema_model = EMAModel(model=self.model, power=0.75)  # 初始化EMA模型
#         self.noise_scheduler = self._create_noise_scheduler()  # 创建噪声调度器
#         self._load_action_stats()
#         self.num_samples = self.config.get("num_samples", 30)

#     def _create_model(self) -> torch.nn.Module:
#         """构建NoMaD模型结构，根据配置选择不同的视觉编码器并验证必需参数"""
#         model_config = self.config
#         vision_encoder_type = model_config.get("vision_encoder", "nomad_vint")  # 默认类型

#         # 根据视觉编码器类型确定必需参数
#         required_params = [
#             "encoding_size",
#             "context_size",
#             "mha_num_attention_heads",
#             "mha_num_attention_layers",
#             "mha_ff_dim_factor",
#             "down_dims",
#             "cond_predict_scale",
#             ]
#         if vision_encoder_type in ["nomad_vint", "vib"]:
#             required_params += [
#                 "mha_num_attention_heads",
#                 "mha_num_attention_layers",
#                 "mha_ff_dim_factor",
#             ]
#         elif vision_encoder_type == "vit":
#             required_params += [
#                 "image_size",
#                 "patch_size",
#                 "mha_num_attention_heads",
#                 "mha_num_attention_layers",
#             ]
#         else:
#             raise ValueError(f"不支持的视觉编码器类型: {vision_encoder_type}")

#         # 构建视觉编码器
#         if vision_encoder_type == "nomad_vint":
#             vision_encoder = NoMaD_ViNT(
#                 obs_encoding_size=model_config["encoding_size"],
#                 context_size=model_config["context_size"],
#                 mha_num_attention_heads=model_config["mha_num_attention_heads"],
#                 mha_num_attention_layers=model_config["mha_num_attention_layers"],
#                 mha_ff_dim_factor=model_config["mha_ff_dim_factor"],
#             )
#         elif vision_encoder_type == "vib":
#             vision_encoder = ViB(
#                 obs_encoding_size=model_config["encoding_size"],
#                 context_size=model_config["context_size"],
#                 mha_num_attention_heads=model_config["mha_num_attention_heads"],
#                 mha_num_attention_layers=model_config["mha_num_attention_layers"],
#                 mha_ff_dim_factor=model_config["mha_ff_dim_factor"],
#             )
#         elif vision_encoder_type == "vit":
#             vision_encoder = ViT(
#                 obs_encoding_size=model_config["encoding_size"],
#                 context_size=model_config["context_size"],
#                 image_size=model_config["image_size"],
#                 patch_size=model_config["patch_size"],
#                 mha_num_attention_heads=model_config["mha_num_attention_heads"],
#                 mha_num_attention_layers=model_config["mha_num_attention_layers"],
#             )
#         vision_encoder = replace_bn_with_gn(vision_encoder)

#         # 构建噪声预测网络和距离预测头
#         noise_pred_net = ConditionalUnet1D(
#             input_dim=2,
#             global_cond_dim=model_config["encoding_size"],
#             down_dims=model_config["down_dims"],
#             cond_predict_scale=model_config["cond_predict_scale"],
#         )
#         dist_pred_net = DenseNetwork(embedding_dim=model_config["encoding_size"])

#         # 整合完整模型
#         model = NoMaD(
#             vision_encoder=vision_encoder,
#             noise_pred_net=noise_pred_net,
#             dist_pred_net=dist_pred_net,
#         ).to(self.device)
#         return model

#     def _create_noise_scheduler(self):
#         """创建扩散噪声调度器"""
#         return DDPMScheduler(
#             num_train_timesteps=self.config.get("num_diffusion_iters", 8),
#             beta_schedule='squaredcos_cap_v2',
#             clip_sample=True,
#             prediction_type='epsilon'
#         )

#     def _create_loggers(self) -> Dict[str, Logger]:
#         """扩散模型特有日志项"""
#         return {
#             "total_loss": Logger("total_loss", "train"),
#             "diffusion_loss": Logger("diffusion_loss", "train"),
#             "dist_loss": Logger("dist_loss", "train"),
#             "uc_action_loss": Logger("uc_action_loss", "train"),
#             "gc_dist_loss": Logger("gc_dist_loss", "train"),
#             # 其他指标...
#         }

#     def _create_optimizer(self) -> torch.optim.Optimizer:
#         """根据配置创建优化器"""
#         optimizer_type = self.config.get("optimizer", "adam").lower()
#         lr = float(self.config.get("lr", 1e-4))
#         weight_decay = self.config.get("weight_decay", 0)

#         param_groups = [
#             {'params': self.model.parameters()},
#             # 可扩展参数分组逻辑
#         ]

#         if optimizer_type == "adam":
#             return Adam(
#                 param_groups,
#                 lr=lr,
#                 betas=tuple(self.config.get("adam_betas", (0.9, 0.98))),
#                 weight_decay=weight_decay
#             )
#         elif optimizer_type == "adamw":
#             return AdamW(param_groups, lr=lr, weight_decay=weight_decay)
#         elif optimizer_type == "sgd":
#             return SGD(
#                 param_groups,
#                 lr=lr,
#                 momentum=self.config.get("momentum", 0.9),
#                 weight_decay=weight_decay
#             )
#         raise ValueError(f"Unsupported optimizer: {optimizer_type}")

#     def _create_scheduler(self, optimizer: torch.optim.Optimizer):
#         """根据配置创建学习率调度器"""
#         scheduler_type = self.config.get("scheduler", "").lower()
#         if not scheduler_type:
#             return None

#         # 基础调度器
#         if scheduler_type == "cosine":
#             return CosineAnnealingLR(
#                 optimizer,
#                 T_max=self.config.get("epochs", 100)
#             )
#         elif scheduler_type == "cyclic":
#             return CyclicLR(
#                 optimizer,
#                 base_lr=self.config["lr"] / 10.0,
#                 max_lr=self.config["lr"],
#                 step_size_up=self.config.get("cyclic_period", 50) // 2,
#                 cycle_momentum=False
#             )
#         elif scheduler_type == "plateau":
#             return ReduceLROnPlateau(
#                 optimizer,
#                 factor=self.config.get("plateau_factor", 0.1),
#                 patience=self.config.get("plateau_patience", 10),
#                 verbose=True
#             )
        
#         # Warmup包装
#         if self.config.get("warmup", False):
#             return GradualWarmupScheduler(
#                 optimizer,
#                 multiplier=1,
#                 total_epoch=self.config.get("warmup_epochs", 5),
#                 after_scheduler=self._create_base_scheduler(optimizer)
#             )
#         return None

#     def _preprocess_batch(self, batch: tuple) -> tuple:
#         """NoMaD数据预处理"""
#         (obs_img, goal_img, actions, dist, goal_pos, _, action_mask, _) = batch
        
#         # 图像处理
#         obs_imgs = torch.split(obs_img, 3, dim=1)
#         obs_tensor = torch.cat([self.transform(img) for img in obs_imgs], dim=1).to(self.device)
#         goal_tensor = self.transform(goal_img).to(self.device)
        
#         # 动作归一化
#         deltas = self._get_delta(actions)
#         ndeltas = self._normalize_data(deltas, self.ACTION_STATS)
        
#         return (
#             obs_tensor,
#             goal_tensor,
#             from_numpy(ndeltas).to(self.device),
#             dist.float().to(self.device),
#             action_mask.to(self.device),
#             actions.to(self.device),  # 原始动作用于评估
#             (obs_imgs[-1], goal_img, goal_pos)  # 可视化数据
#         )

#     def _train_step(self, batch: tuple, epoch, batch_idx: int) -> Dict[str, torch.Tensor]:
#         """扩散模型训练步骤"""
#         obs_tensor, goal_tensor, naction, dist_label, action_mask, true_actions, viz_data = self._preprocess_batch(batch)

#         self.pred_horizon = getattr(self, "pred_horizon", true_actions.shape[1])
#         self.action_dim = getattr(self, "action_dim", true_actions.shape[2])

#         # 生成随机目标掩码
#         B = obs_tensor.size(0)
#         goal_mask = (torch.rand(B) < self.config["goal_mask_prob"]).long().to(self.device)
        
#         # 前向传播
#         obs_cond = self.model.vision_encoder(obs_tensor, goal_tensor, goal_mask)
#         noise_pred, dist_pred = self._forward_diffusion(naction, obs_cond)
        
#         # 损失计算
#         losses = self._compute_train_losses(noise_pred, dist_pred, naction, dist_label, action_mask, goal_mask)
        
#         # 反向传播
#         self.optimizer.zero_grad()
#         losses["total_loss"].backward()
#         self.optimizer.step()
        
#         # EMA更新
#         self.ema_model.step(self.model)
        
#         if self._should_visualize(batch_idx, self.config["image_log_freq"]):
#             # 解包可视化数据
#             viz_obs, viz_goal, viz_goal_pos = viz_data  # 新增解包语句
#             self._visualize_diffusion_actions(
#                 batch_obs_images=obs_tensor,
#                 batch_goal_images=goal_tensor,
#                 batch_viz_obs_images=viz_obs,
#                 batch_viz_goal_images=viz_goal,
#                 batch_action_label=true_actions,
#                 batch_distance_labels=dist_label,
#                 batch_goal_pos=viz_goal_pos,
#                 eval_type="train",
#                 epoch=epoch,
#                 num_samples=self.num_samples,
#             )

#         return losses

#     def _forward_diffusion(self, naction, obs_cond):
#         """扩散过程前向传播"""

#         noise = torch.randn_like(naction)
#         timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (naction.size(0),), device=self.device)
#         noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)

#         return self.model.noise_pred_net(sample=noisy_action, timestep=timesteps, global_cond=obs_cond), self.model.dist_pred_net(obs_cond)

#     def _compute_train_losses(self, noise_pred, dist_pred, naction, dist_label, action_mask, goal_mask):
#         """训练阶段损失计算"""
#         # 扩散损失（带掩码）
#         def action_reduce(loss):
#             while loss.dim() > 1:
#                 loss = loss.mean(dim=-1)
#             return (loss * action_mask).mean() / (action_mask.mean() + 1e-2)
        
#         diffusion_loss = action_reduce(F.mse_loss(noise_pred, naction, reduction='none'))
        
#         # 距离预测损失
#         dist_loss = F.mse_loss(dist_pred.squeeze(), dist_label)
#         dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / ((1 - goal_mask.float()).mean() + 1e-2)
        
#         return {
#             "total_loss": self.config["alpha"] * dist_loss + (1 - self.config["alpha"]) * diffusion_loss,
#             "diffusion_loss": diffusion_loss,
#             "dist_loss": dist_loss
#         }

#     def _update_loggers(self, losses: dict):
#         """更新训练日志"""
#         for key in ["total_loss", "diffusion_loss", "dist_loss"]:
#             self.loggers[key].log_data(losses[key].item())

#     def _eval_step(self, batch: tuple, epoch: int, batch_idx: int, eval_type: str) -> Dict[str, torch.Tensor]:
#         """评估步骤扩展实现"""
#         model = self.ema_model.averaged_model
#         obs_tensor, goal_tensor, _, dist_label, action_mask, true_actions, viz_data = self._preprocess_batch(batch)

#         self.pred_horizon = getattr(self, "pred_horizon", true_actions.shape[1])
#         self.action_dim = getattr(self, "action_dim", true_actions.shape[2])

#         with torch.no_grad():
#             # 生成预测结果

#             pred_dict = self._generate_diffusion_output(obs_tensor, goal_tensor, 1)
            
#             # 计算所有指标
#             metrics = self._compute_eval_metrics(
#                 pred_dict['uc_actions'],
#                 pred_dict['gc_actions'],
#                 pred_dict['gc_distance'],
#                 true_actions,
#                 dist_label,
#                 action_mask
#             )

#             # 三种掩码条件下的扩散损失
#             mask_types = ['random', 'no_mask', 'full_mask']
#             for mask_name in mask_types:
#                 mask = self._generate_mask(mask_name, obs_tensor.size(0))
#                 metrics.update(self._compute_mask_loss(obs_tensor, goal_tensor, mask, action_mask, true_actions, mask_name))

#             if self._should_visualize(batch_idx, self.config["image_log_freq"]):
#                 # 解包可视化数据
#                 viz_obs, viz_goal, viz_goal_pos = viz_data  # 新增解包语句
#                 self._visualize_diffusion_actions(
#                     batch_obs_images=obs_tensor,
#                     batch_goal_images=goal_tensor,
#                     batch_viz_obs_images=viz_obs,
#                     batch_viz_goal_images=viz_goal,
#                     batch_action_label=true_actions,
#                     batch_distance_labels=dist_label,
#                     batch_goal_pos=viz_goal_pos,
#                     eval_type=eval_type,
#                     epoch=epoch,
#                     num_samples=self.num_samples,
#                 )

#         return metrics

#     def _update_eval_loggers(self, losses: dict, eval_type: str):
#         """修正方法名调用"""
#         for key in losses:
#             metric_name = f"{eval_type}/{key}"
#             if metric_name not in self.loggers:
#                 self.loggers[metric_name] = Logger(key, eval_type)
#             # 将update改为log_data
#             self.loggers[metric_name].log_data(losses[key].item())

#     def _compute_additional_metrics(self, batch: tuple) -> dict:
#         """整合_compute_losses_nomad功能"""
#         # 解包数据
#         obs_tensor, goal_tensor, _, dist_label, action_mask, true_actions, _ = self._preprocess_batch(batch)

#         self.pred_horizon = getattr(self, "pred_horizon", true_actions.shape[1])
#         self.action_dim = getattr(self, "action_dim", true_actions.shape[2])

#         # 生成预测动作
#         pred_dict = self._generate_diffusion_output(
#             obs_images=obs_tensor,
#             goal_images=goal_tensor,
#             num_samples=self.num_samples
#         )
        
#         # 计算基础损失
#         def action_reduce(loss_tensor):
#             while loss_tensor.dim() > 1:
#                 loss_tensor = loss_tensor.mean(dim=-1)
#             return (loss_tensor * action_mask).mean() / (action_mask.mean() + 1e-2)

#         # 动作损失计算
#         uc_action_loss = action_reduce(F.mse_loss(pred_dict['uc_actions'], true_actions, reduction='none'))
#         gc_action_loss = action_reduce(F.mse_loss(pred_dict['gc_actions'], true_actions, reduction='none'))
        
#         # 轨迹余弦相似度
#         def compute_cosine_sim(pred, gt):
#             flat_pred = pred[:, :, :2].flatten(1)
#             flat_gt = gt[:, :, :2].flatten(1)
#             return F.cosine_similarity(flat_pred, flat_gt, dim=-1).mean()

#         return {
#             "uc_action_loss": uc_action_loss,
#             "gc_action_loss": gc_action_loss,
#             "gc_dist_loss": F.mse_loss(pred_dict['gc_distance'], dist_label.unsqueeze(-1)),
#             "traj_cosine_sim": compute_cosine_sim(pred_dict['gc_actions'], true_actions)
#         }

#     def _generate_diffusion_output(self, obs_images: torch.Tensor, goal_images: torch.Tensor, num_samples: int) -> dict:
#         """实现model_output功能"""
#         model = self.ema_model.averaged_model
        
#         # 无目标条件生成
#         uc_actions = self._denoise_loop(
#             model, obs_images, goal_images,
#             mask=torch.ones(obs_images.size(0)).long().to(self.device),
#             num_samples=num_samples,
#         )
        
#         # 全目标条件生成
#         gc_actions = self._denoise_loop(
#             model, obs_images, goal_images,
#             mask=torch.zeros(obs_images.size(0)).long().to(self.device),
#             num_samples=num_samples,
#         )
        
#         # 距离预测
#         obsgoal_cond = model.vision_encoder(obs_images, goal_images, 
#                                            torch.zeros(obs_images.size(0)).long().to(self.device))
#         obsgoal_cond.repeat_interleave(num_samples, dim=0)
#         gc_distance = model.dist_pred_net(obsgoal_cond)
        
#         return {
#             'uc_actions': self._get_action(uc_actions, self.ACTION_STATS),
#             'gc_actions': self._get_action(gc_actions, self.ACTION_STATS),
#             'gc_distance': gc_distance
#         }

#     def _denoise_loop(self, model, obs, goal, mask: torch.Tensor, num_samples) -> torch.Tensor:
#         """逆向扩散核心逻辑"""

#         # 获取条件向量
#         cond = model.vision_encoder(obs, goal, mask)
#         cond = cond.repeat_interleave(num_samples, dim=0)  # 支持num_samples扩展

#         # 初始化噪声动作
#         noisy_actions = torch.randn(
#             (cond.size(0), self.pred_horizon, self.action_dim), 
#             device=self.device
#         )

#         # 迭代去噪
#         for t in self.noise_scheduler.timesteps:
#             noise_pred = model.noise_pred_net(
#                 sample=noisy_actions,
#                 timestep=torch.full((noisy_actions.size(0),), t, device=self.device),
#                 global_cond=cond
#             )
#             noisy_actions = self.noise_scheduler.step(
#                 model_output=noise_pred,
#                 timestep=t,
#                 sample=noisy_actions
#             ).prev_sample
            
#         return noisy_actions

#     def _compute_eval_metrics(self, uc_actions, gc_actions, gc_dist, true_actions, dist_label, action_mask):
#         """实现_compute_losses_nomad功能"""
#         def action_reduce(loss):
#             while loss.dim() > 1:
#                 loss = loss.mean(dim=-1)
#             return (loss * action_mask).mean() / (action_mask.mean() + 1e-2)
        
#         # 动作损失
#         uc_action_loss = action_reduce(F.mse_loss(uc_actions, true_actions, reduction='none'))
#         gc_action_loss = action_reduce(F.mse_loss(gc_actions, true_actions, reduction='none'))
        
#         # 轨迹相似度
#         def flatten_cosine(a, b):
#             return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1).mean()
        
#         return {
#             "uc_action_loss": uc_action_loss,
#             "gc_action_loss": gc_action_loss,
#             "gc_dist_loss": F.mse_loss(gc_dist, dist_label.unsqueeze(-1)),
#             "waypoint_cos_sim": action_reduce(
#                 F.cosine_similarity(gc_actions[..., :2], true_actions[..., :2], dim=-1)),
#             "trajectory_cos_sim": flatten_cosine(gc_actions[..., :2], true_actions[..., :2])
#         }

#     def _generate_mask(self, mask_type: str, batch_size: int) -> torch.Tensor:
#         """生成评估用掩码"""
#         if mask_type == 'random':
#             return (torch.rand(batch_size) < self.config["goal_mask_prob"]).long().to(self.device)
#         elif mask_type == 'no_mask':
#             return torch.zeros(batch_size, dtype=torch.long).to(self.device)
#         elif mask_type == 'full_mask':
#             return torch.ones(batch_size, dtype=torch.long).to(self.device)
#         raise ValueError(f"未知掩码类型: {mask_type}")

#     def _compute_mask_loss(self, obs, goal, mask, action_mask, actions, mask_name):
#         """计算指定掩码条件下的损失"""
#         # cond = self.ema_model.averaged_model.vision_encoder(obs, goal, mask)
#         def action_reduce(loss):
#             while loss.dim() > 1:
#                 loss = loss.mean(dim=-1)
#             return (loss * action_mask).mean() / (action_mask.mean() + 1e-2)
#         model = self.ema_model.averaged_model
#         pred_dict = self._denoise_loop(model, obs, goal, mask, 1)
#         pred_dict = self._get_action(pred_dict, self.ACTION_STATS)
#         return {
#             f"{mask_name}_loss": action_reduce(F.mse_loss(pred_dict, actions, reduction='none'))
#         }


#     def _visualize_diffusion_actions(
#         self,
#         batch_obs_images: torch.Tensor,
#         batch_goal_images: torch.Tensor,
#         batch_viz_obs_images: torch.Tensor,
#         batch_viz_goal_images: torch.Tensor,
#         batch_action_label: torch.Tensor,
#         batch_distance_labels: torch.Tensor,
#         batch_goal_pos: torch.Tensor,
#         eval_type: str,
#         epoch: int,
#         num_images_log: int = 8,
#         num_samples: int = None, 
#     ):
#         if num_samples is None:
#             num_samples = self.num_samples

#         """封装后的可视化方法"""
#         # 创建可视化目录
#         visualize_path = self._create_visualization_path(eval_type, epoch)
        
#         # 限制日志记录数量
#         num_images_log = self._adjust_logging_quantity(num_images_log, batch_obs_images)

#         # 分割批次避免内存溢出
#         batch_chunks = self._split_into_chunks(
#             batch_obs_images, batch_goal_images, 
#             batch_viz_obs_images, batch_viz_goal_images,
#             batch_action_label, batch_goal_pos,
#             chunk_size=num_images_log,
#         )
        
#         chunk = batch_chunks[0]

#         pred_dict = self._generate_visualization_predictions(chunk, num_samples)
        
#         # 绘制并保存图表
#         self._plot_and_save_figures(
#             chunk, pred_dict, 
#             batch_distance_labels, 
#             visualize_path,
#             num_samples,
#         )

#     def _create_visualization_path(self, eval_type: str, epoch: int) -> str:
#         """创建可视化存储路径"""
#         visualize_path = os.path.join(
#             self.project_folder,
#             "visualize",
#             eval_type,
#             f"epoch{epoch}",
#             "action_sampling_prediction",
#         )
#         if not os.path.isdir(visualize_path):
#             os.makedirs(visualize_path)
#         return visualize_path

#     def _adjust_logging_quantity(self, num_images_log: int, batch_obs_images: torch.Tensor) -> int:
#         """调整实际记录数量"""
#         return min(num_images_log, batch_obs_images.shape[0])

#     def _split_into_chunks(self, *batch_data, chunk_size: int = 8):
#         """安全分块方法"""
#         # 获取实际批量大小
#         batch_size = batch_data[0].shape[0] if isinstance(batch_data[0], (torch.Tensor, np.ndarray)) else len(batch_data[0])

#         # 统一数据转换
#         processed_data = []
#         for data in batch_data:
#             # 确保所有数据维度对齐
#             assert len(data) == batch_size, f"数据维度不匹配: 预期 {batch_size}, 实际 {len(data)}"
#             processed_data.append(data)

#         # 安全分块
#         return [
#             tuple(data[i:i+chunk_size] for data in processed_data)
#             for i in range(0, batch_size, chunk_size)
#         ]

#     def _generate_visualization_predictions(self, batch_chunk, num_samples: int):
#         """生成模型预测结果"""
#         obs, goal, viz_obs, viz_goal, actions, goal_pos = batch_chunk

#         # 无约束预测
#         uc_actions = self._denoise_loop(
#             self.ema_model.averaged_model,
#             obs.to(self.device),
#             goal.to(self.device),
#             mask=torch.ones(len(obs)).to(self.device),
#             num_samples=num_samples,
#         )
#         uc_actions = self._get_action(uc_actions, self.ACTION_STATS)

#         # 目标条件预测
#         gc_actions = self._denoise_loop(
#             self.ema_model.averaged_model,
#             obs.to(self.device),
#             goal.to(self.device),
#             mask=torch.zeros(len(obs)).to(self.device),
#             num_samples=num_samples,
#         )
#         gc_actions = self._get_action(gc_actions, self.ACTION_STATS)

#         # 距离预测
#         obsgoal_cond = self.ema_model.averaged_model.vision_encoder(obs, goal, 
#                                            torch.zeros(len(obs)).long().to(self.device))
#         obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)
#         gc_distance = self.ema_model.averaged_model.dist_pred_net(obsgoal_cond)

#         return {
#             'uc_actions': to_numpy(uc_actions),
#             'gc_actions': to_numpy(gc_actions),
#             'viz_obs': to_numpy(viz_obs),
#             'viz_goal': to_numpy(viz_goal),
#             'goal_pos': to_numpy(goal_pos),
#             'gc_distance': to_numpy(gc_distance),
#             'action_labels': to_numpy(actions),
#         }
#     def _plot_and_save_figures(self, batch_chunk, pred_dict, distance_labels, save_dir: str, num_samples: int):
#         """集成式可视化绘图方法"""
#         wandb_images = []
#         os.makedirs(save_dir, exist_ok=True)
#         # 获取批量大小和样本索引范围
#         batch_size = len(batch_chunk[0])
#         viz_indices = range(batch_size)
        
#         # 合并轨迹数据生成与绘图
#         for idx in viz_indices:
#             fig = self._generate_single_plot(idx, pred_dict, distance_labels, num_samples)
#             save_path = os.path.join(save_dir, f"sample_{idx}.png")
#             plt.savefig(save_path, bbox_inches='tight', dpi=150)
#             plt.close(fig)

#     def _generate_single_plot(self, index: int, pred_dict: dict, distance_labels, num_samples: int):
#         """单样本全要素绘图"""
#         fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        
#         # --- 轨迹绘制 ---
#         traj_data = self._prepare_trajectory_data(index, pred_dict, num_samples)
#         self._plot_trajectory(axs[0], traj_data, pred_dict['goal_pos'][index])
        
#         # --- 图像绘制 ---
#         self._plot_observation_image(axs[1], pred_dict['viz_obs'][index])

#         self._plot_goal_image(axs[2], pred_dict['viz_goal'][index], 
#                             distance_labels[index], 
#                             pred_dict['gc_distance'][index * num_samples : (index + 1) * num_samples])
        
#         plt.tight_layout()
#         return fig

#     def _prepare_trajectory_data(self, index: int, pred_dict: dict, num_samples: int) -> np.ndarray:
#         """轨迹数据预处理"""
#         uc_actions = pred_dict['uc_actions'][index * num_samples : (index + 1) * num_samples]  # [num_samples, pred_horizon, 2]
#         gc_actions = pred_dict['gc_actions'][index * num_samples : (index + 1) * num_samples]  # [num_samples, pred_horizon, 2]
#         gt_action = pred_dict['action_labels'][index]  # [pred_horizon, 2]

#         # 维度对齐 (添加样本维度)
#         return np.concatenate([
#             uc_actions,
#             gc_actions,
#             gt_action[None, ...]  # 升维至 [1, pred_horizon, 2]
#         ], axis=0)

#     def _plot_trajectory(self, ax: plt.Axes, traj_data: np.ndarray, goal_pos: np.ndarray):
#         """核心轨迹绘制方法"""
#         # 动态颜色配置
#         num_samples = traj_data.shape[0] - 1  # 最后一个是GT
#         colors = (
#             ["#FF000080"] * num_samples +  # 半透明红色
#             ["#00FF0080"] * num_samples +  # 半透明绿色
#             ["m"]                          # 品红色GT
#         )
        
#         plot_trajs_and_points(
#             ax=ax,
#             list_trajs=traj_data,
#             list_points=[np.zeros(2), goal_pos],
#             traj_colors=colors,
#             point_colors=["g", "r"],
#             traj_labels=None,
#             point_labels=None,
#             traj_alphas=[0.3]*(2*num_samples) + [1.0],
#             quiver_freq=3
#         )
#         ax.set(xlabel="X (m)", ylabel="Y (m)", title="Action Distribution")

#     def _plot_observation_image(self, ax: plt.Axes, obs_img: np.ndarray):
#         """观测图像绘制优化"""
#         if obs_img.shape[0] == 3:  # 处理通道顺序
#             obs_img = np.moveaxis(obs_img, 0, -1)
#         ax.imshow(obs_img, interpolation='hanning')
#         ax.axis('off')
#         ax.set_title("Current Observation", pad=10)

#     def _plot_goal_image(self, ax: plt.Axes, 
#                         goal_img: np.ndarray, 
#                         label_dist: float, 
#                         pred_actions: np.ndarray):
#         """目标图像带统计信息"""
#         # 计算预测统计量

#         pred_dist = np.linalg.norm(pred_actions[-1, :2], axis=0).mean()
#         std_dist = np.linalg.norm(pred_actions[-1, :2], axis=0).std()
        
#         # 绘制图像
#         if goal_img.shape[0] == 3:
#             goal_img = np.moveaxis(goal_img, 0, -1)
#         ax.imshow(goal_img, cmap='viridis')
#         ax.axis('off')
        
#         ax.set_title(f"goal: label={label_dist:.2f}m gc_dist={pred_dist:.2f}±{std_dist:.2f}m")

#     def _should_visualize(self, step: int, log_freq: int) -> bool:
#         """判断是否需要可视化"""
#         return log_freq > 0 and step % log_freq == 0