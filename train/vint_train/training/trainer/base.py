import itertools
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    LRScheduler,
)
from torchvision import transforms
from tqdm.auto import tqdm  # 自动适配Jupyter/终端环境
import yaml

from vint_train.visualizing.visualize_utils import from_numpy, to_numpy

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
        print_freq = self.logging_config['print_freq']
        wandb_freq = self.logging_config['wandb_freq']

        # 控制台日志
        if print_freq > 0 and batch_idx % print_freq == 0:
            self._print_batch_metrics(epoch, batch_idx, total_batches, mode)

        # WandB日志
        if self.logging_config['use_wandb'] and wandb_freq > 0 and batch_idx % wandb_freq == 0:
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
    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler:
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
        return freq > 0 and (batch_idx % freq == 0) and (self.config["num_images_log"] > 0)

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
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """统一日志记录"""
        if self.config["use_wandb"]:
            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)
            
    def _visualize_results(self, batch: Any, epoch: int, mode: str):
        """统一可视化接口"""
        pass
