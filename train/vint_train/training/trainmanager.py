import wandb
import os
import numpy as np
import inspect
from abc import ABC, abstractmethod
from typing import Dict, Optional
from prettytable import PrettyTable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel

from typing import *
import inspect
from dataclasses import dataclass, fields

from vint_train.training.trainer import *
from vint_train.training.logger import *
# --------------------------
# 核心抽象层
# --------------------------

# training_params.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
from torch import nn
import torch

class ParamRegister:
    """参数元数据注册中心"""
    
    _param_metadata = {
        'image_size': {
            'type': list,
            'default': [96, 96]
        },
        'alpha': {
            'type': float,
            'default': 0.5
        }
    }

    @classmethod
    def validate_params(cls, params: Dict) -> Dict:
        validated = {}
        for name, val in params.items():
            if name in cls._param_metadata:
                meta = cls._param_metadata[name]
                # 新增类型转换逻辑
                try:
                    val = meta['type'](val)  # 尝试强制类型转换
                except (TypeError, ValueError) as e:
                    raise TypeError(f"Cannot cast {name} to {meta['type']}: {str(e)}")
                # 类型检查
                if not isinstance(val, meta['type']):
                    raise TypeError(...)
                validated[name] = val
            else:
                validated[name] = val
        
        # 填充默认值
        for name, meta in cls._param_metadata.items():
            if name not in validated:
                validated[name] = meta['default']
                
        return validated

@dataclass
class TrainingParams:
    """ 基础训练参数（所有模型通用） """
    # 必需参数
    config: Dict[str, Any]
    train_loader: DataLoader
    device: torch.device
    # 可选参数
    test_dataloaders: Optional[Dict[str, DataLoader]] = None
    transform: Optional[Any] = None
    current_epoch: int = 0
    
    # 扩展参数（模型专用）
    noise_scheduler: Optional[Any] = None
    diffusion: Optional[Any] = None
    goal_mask_prob: float = None
    alpha: float = 0.5

    # 配置参数提升映射表（配置键 -> 实例属性）
    __CONFIG_PROMOTIONS__ = {
        'goal_mask_prob': 'goal_mask_prob',  # 直接提升同名参数
        'normalize': 'normalized',
    }

    @property
    def model_type(self) -> str:
        return self.config.get("model_type", "unknown")
    
    def __post_init__(self):
        """后初始化处理"""
        self._promote_config()
        self._process_special_types()
        # self.validate()

    def _promote_config(self):
        """智能提升配置参数到顶层属性"""
        for config_path, attr_name in self.__CONFIG_PROMOTIONS__.items():
            # 支持多级配置路径解析（如"training.batch_size"）
            keys = config_path.split('.')
            value = self.config
            try:
                for key in keys:
                    value = value[key]
            except (KeyError, TypeError):
                continue
            
            # 类型安全转换
            if attr_name == 'device':
                value = torch.device(value) if isinstance(value, str) else value
            
            setattr(self, attr_name, value)

    def _process_special_types(self):
        """处理需要特殊类型转换的字段"""
        # 自动转换设备类型
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    @property
    def promoted_params(self) -> dict:
        """返回实际生效的提升参数"""
        return {
            attr: getattr(self, attr) 
            for _, attr in self.__CONFIG_PROMOTIONS__.items()
            if hasattr(self, attr)
        }

class TrainingStrategy(ABC):
    """训练策略抽象基类"""
    
    def __init__(self, context):
        self.ctx = context
        self.trainer = None
        
    @abstractmethod
    def execute_training_epoch(self, epoch: int):
        pass
    
    @abstractmethod
    def execute_evaluation(self, epoch: int):
        pass
    
    def save_checkpoint(self, epoch: int):
        """通用保存逻辑"""
        CheckpointManager.save(
            model=self.trainer.model,
            path=self.ctx.project_folder,
            ema_model=self.trainer.ema_model,
            optimizer=self.trainer.optimizer,
            scheduler=self.trainer.scheduler,
            epoch=epoch
        )

# --------------------------
# 具体策略实现
# --------------------------

class UniversalTrainingStrategy(TrainingStrategy):
    """通用训练策略，通过命名约定自动匹配函数"""
    
    def __init__(self, context):
        super().__init__(context)
        self._init_components()
        self.trainer = self._init_trainer()

    def _init_components(self):
        """动态初始化Trainer"""
        model_type = self.ctx.model_type
        trainer_cls = self._get_trainer_class(model_type)

        # 初始化Trainer
        self.trainer = trainer_cls(self.ctx.params)

    def _init_trainer(self) -> 'BaseTrainer':
        """动态初始化Trainer"""
        trainer_class = self._discover_trainer()
        return trainer_class(self.ctx.params)
    
    def _discover_trainer(self) -> type:
        """通过模型类型发现对应Trainer"""
        model_type = self.ctx.model_type.lower()
        for cls in BaseTrainer.__subclasses__():
            if cls.__name__.lower().startswith(model_type):
                return cls
        raise ValueError(f"No trainer found for model type: {self.ctx.model_type}")

    def _get_trainer_class(self, model_type: str) -> Type[BaseTrainer]:
        """自动化匹配Trainer类"""
        target_name = f"{model_type}Trainer".lower()
        
        try:
            # 遍历所有BaseTrainer子类
            for cls in BaseTrainer.__subclasses__():
                cls_name = cls.__name__.lower()
                
                if cls_name == target_name or \
                cls_name.replace("_", "") == target_name:
                    trainer_cls = cls
                    break
        except:
            # 未找到时显示可用类列表
            available = [c.__name__ for c in BaseTrainer.__subclasses__()]
            raise ValueError(
                f"No trainer found for '{model_type}'. "
                f"Available trainers: {available}\n"
                f"Naming rule: {{model_type}}Trainer (case-insensitive)"
            )

        return trainer_cls

    def _bind_components(self):
        """将上下文组件绑定到Trainer"""
        self.trainer.device = self.ctx.device

    def _resolve_function(self, name: str, fallback: Optional[str] = None) -> Optional[Callable]:
        """解析目标函数，支持回退机制"""
        func = globals().get(name)
        if not func and fallback:
            func = globals().get(fallback)
        return func
    
    def _prepare_arguments(self, func: Callable, exclude: list = None) -> dict:
        """智能准备函数参数"""
        sig = inspect.signature(func)
        params = {}

        # 优先使用提升后的参数
        arg_pool = {
            **self.ctx.promoted_params,  # 包含提升后的配置参数
            **vars(self.ctx),            # 原始参数
            "epoch": self.ctx.config.get("current_epoch", 0),
            "config": self.ctx.config,
            "transform": self.ctx.transform,
        }

        for param in sig.parameters.values():
            # 处理参数别名
            if param.name in arg_pool:
                params[param.name] = arg_pool[param.name]
            elif param.name in self.ctx.config:  # 从config中提取参数
                params[param.name] = self.ctx.config[param.name]
        
        # 排除不需要的参数
        if exclude:
            params = {k: v for k, v in params.items() if k not in exclude}
            
        return params
    
    def execute_training_epoch(self, epoch: int):
        """执行训练epoch"""
        print(f"Start {self.ctx.model_type} Training Epoch {epoch}")
        self.trainer.train_epoch(self.ctx.train_loader, epoch)

    def execute_evaluation(self, epoch: int):
        """执行评估"""
        if not self._should_evaluate(epoch):
            return
            
        print(f"Start {self.ctx.model_type} Evaluation Epoch {epoch}")
        for dataset_type, loader in self.ctx.test_loaders.items():
            eval_metrics = self.trainer.evaluate(loader, epoch, dataset_type)
    
    def _should_evaluate(self, epoch: int) -> bool:
        """智能判断评估条件"""
        eval_freq = self.ctx.params.get("eval_freq", 1)
        return (epoch + 1) % eval_freq == 0

# --------------------------
# 上下文管理
# --------------------------
class TrainingContext:
    """统一训练上下文"""

    DEFAULT_CONFIG = {
        "current_epoch": 0,
        "epochs": 100,
        "eval_freq": 1,
        "train": True,
        "project_folder": "./checkpoints",
        "use_wandb": False
    }

    CORE_PARAMS = {
        'project_folder', 
        'use_wandb',
    }  # 核心参数白名单

    def __init__(self, params: TrainingParams):

        # 参数结构解构
        self._raw_params = params.__dict__
        
        # 参数分类处理
        self._init_core_params(params)     # 核心参数初始化
        self._init_extension_params()     # 扩展参数池
        self._init_components(params)     # 组件初始化

    def _init_core_params(self, params: 'TrainingParams'):
        """初始化核心参数"""
        # 解构核心参数
        self.device = getattr(params, 'device', torch.device('cpu'))

        self.project_folder = params.config.get('project_folder', './checkpoints')
        
        # 模型类型推断
        self.model_type = params.config.get('model_type', 'unknown')
        
        # 日志相关
        self.use_wandb = params.config.get('use_wandb', False)

    def _init_extension_params(self):
        """构建扩展参数池"""
        # 从原始参数提取非核心参数
        self.params = {
            k: v for k, v in self._raw_params.items() 
        }
        
        # 合并默认配置
        self.params.update(self._raw_params.get('config', {}))

        # 参数校验与补全
        self.params = ParamRegister.validate_params(self.params)

    def _init_components(self, params: 'TrainingParams'):
        """初始化系统组件"""
        # 数据加载器
        self.train_loader = params.train_loader
        self.test_loaders = params.test_dataloaders
        
        # 初始化策略系统
        self.strategy = UniversalTrainingStrategy(self)
        self.model = self.strategy.trainer._create_model()

        self._init_logging_system()

    def _init_logging_system(self):
        """初始化分层日志系统"""
        # 核心日志配置
        self.log_config = {
            'use_wandb': self.use_wandb,
            'project_folder': self.project_folder,
            'image_log_freq': self.get_param('image_log_freq', 100),
            'metric_log_freq': self.get_param('metric_log_freq', 10)
        }
        
        # 创建日志路由器
        self.log_router = LogRouter(
            config=self.log_config,
            model_type=self.model_type
        )

    def get_param(self, name: str, default: Any = None) -> Any:
        """统一参数访问接口"""
        return self.params.get(name, default)

    def _validate_required_components(self):
        """校验关键组件完整性"""
        assert hasattr(self, 'train_loader'), "Missing required component: train_loader"
    
    def run(self):
        """执行完整训练流程"""
        start_epoch = self.params["current_epoch"]
        end_epoch = self.params["epochs"]

        for epoch in range(start_epoch, end_epoch):
            # 训练阶段
            if self.params["train"]:
                self.strategy.execute_training_epoch(epoch)
                
            # 评估阶段
            self.strategy.execute_evaluation(epoch)
                
            # 保存检查点
            self.strategy.save_checkpoint(epoch)
            
            # 更新学习率
            self._update_learning_rate()
            
    def _should_evaluate(self, epoch: int) -> bool:
        return (epoch + 1) % self.params["eval_freq"] == 0
    
    def _update_learning_rate(self):
        """更新学习率"""
        self.strategy.trainer.scheduler.step()

    def get_all_variable_names(self) -> list:
        """获取实例所有成员变量名"""
        return list(vars(self).keys())

    def get_all_method_names(self) -> list:
        """获取类所有方法名（包含私有和公有）"""
        return [name for name, obj in inspect.getmembers(self.__class__) 
                if inspect.isfunction(obj) or inspect.ismethod(obj)]

# --------------------------
# 工具模块
# --------------------------
class CheckpointManager:
    """统一检查点管理"""
    
    @staticmethod
    def save(model, path: str, ema_model=None, optimizer=None, scheduler=None, epoch=None):
        if ema_model is not None:
            numbered_path = os.path.join(path, f"ema_{epoch}.pth")
            torch.save(ema_model.averaged_model.state_dict(), numbered_path)
            numbered_path = os.path.join(path, f"ema_latest.pth")
            print(f"Saved EMA model to {numbered_path}")

        latest_path = os.path.join(path, f"latest.pth")
        numbered_path = os.path.join(path, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # save optimizer
        numbered_path = os.path.join(path, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(path, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        numbered_path = os.path.join(path, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(path, f"scheduler_latest.pth")
        torch.save(scheduler.state_dict(), latest_scheduler_path)

# --------------------------
# 对外接口
# --------------------------
def training(params: TrainingParams):
    """统一训练入口"""
    context = TrainingContext(params)
    context.run()

# --------------------------
# 兼容原有函数
# --------------------------

def load_model(model, model_type, checkpoint: dict) -> None:
    """加载模型（保持兼容）"""
    if model_type in ["nomad", "navibridge"]:
        if "model" in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        if 'ema' in checkpoint:
            EMAModel(model).load_state_dict(checkpoint['ema'])
    else:
        if "model" in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params