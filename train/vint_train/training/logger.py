import numpy as np
import wandb
from abc import ABC, abstractmethod

class Logger:
    def __init__(
        self,
        name: str,
        dataset: str,
        window_size: int = 10,
        rounding: int = 4,
    ):
        """
        Args:
            name (str): Name of the metric
            dataset (str): Name of the dataset
            window_size (int, optional): Size of the moving average window. Defaults to 10.
            rounding (int, optional): Number of decimals to round to. Defaults to 4.
        """
        self.data = []
        self.name = name
        self.dataset = dataset
        self.rounding = rounding
        self.window_size = window_size

    def display(self) -> str:
        latest = round(self.latest(), self.rounding)
        average = round(self.average(), self.rounding)
        moving_average = round(self.moving_average(), self.rounding)
        output = f"{self.full_name()}: {latest} ({self.window_size}pt moving_avg: {moving_average}) (avg: {average})"
        return output

    def log_data(self, data: float):
        if not np.isnan(data):
            self.data.append(data)

    def full_name(self) -> str:
        return f"{self.name} ({self.dataset})"

    def latest(self) -> float:
        if len(self.data) > 0:
            return self.data[-1]
        return np.nan

    def average(self) -> float:
        if len(self.data) > 0:
            return np.mean(self.data)
        return np.nan

    def moving_average(self) -> float:
        if len(self.data) > self.window_size:
            return np.mean(self.data[-self.window_size :])
        return self.average()

    @property
    def avg(self) -> float:
        """提供属性化访问接口"""
        return self.average()

class WandBLoggerAdapter:
    """扩展Logger类以支持WandB"""
    
    def __init__(self, base_logger, use_wandb: bool, project_folder: str):
        self.base_logger = base_logger
        self.use_wandb = use_wandb
        if self.use_wandb:
            import wandb
            wandb.init(project="training_project", dir=project_folder)

    def log_data(self, data: float):
        """增强后的日志方法"""
        self.base_logger.log_data(data)
        
        # if self.use_wandb:
        #     wandb.log({
        #         f"{self.base_logger.name}/{self.base_logger.dataset}": data
        #     })

# 新增日志路由类
class LogRouter:
    """统一管理所有日志渠道"""
    def __init__(self, config: dict, model_type: str):
        self.config = config
        self.model_type = model_type
        self._init_channels()
        
    def _init_channels(self):
        """初始化日志渠道"""
        self.channels = []
        
        # 控制台日志
        self.channels.append(ConsoleLogger())
        
        # WandB集成
        # if self.config['use_wandb']:
        #     self.channels.append(WandBLogger(
        #         project=self.model_type,
        #         save_dir=self.config['project_folder']
        #     ))
            
        # 文件日志（可选）
        # if self.config.get('enable_file_log'):
        #     self.channels.append(FileLogger(
        #         log_dir=os.path.join(self.config['project_folder'], "logs")
        #     ))

    def log_metrics(self, metrics: dict, step: int):
        """路由指标数据"""
        for channel in self.channels:
            if isinstance(channel, MetricLogger):
                channel.log_metrics(metrics, step)

    def log_images(self, images: dict, step: int):
        """路由图像数据"""
        if step % self.config['image_log_freq'] == 0:
            for channel in self.channels:
                if isinstance(channel, ImageLogger):
                    channel.log_images(images, step)

# 基础日志接口
class MetricLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        pass

class ImageLogger(ABC):
    @abstractmethod
    def log_images(self, images: dict, step: int):
        pass

# 具体实现（保持原有Logger不变）
class ConsoleLogger(MetricLogger):
    def log_metrics(self, metrics: dict, step: int):
        for k, v in metrics.items():
            print(f"[Step {step}] {k}: {v:.4f}")

# class WandBLogger(MetricLogger, ImageLogger):
#     def __init__(self, project: str, save_dir: str):
#         import wandb
#         wandb.init(project=project, dir=save_dir)
#         self.wandb = wandb
        
#     def log_metrics(self, metrics: dict, step: int):
#         self.wandb.log(metrics, step=step)
        
#     def log_images(self, images: dict, step: int):
#         wandb_images = {}
#         for tag, tensor in images.items():
#             if tensor.dim() == 4:  # 图像批次
#                 for i in range(min(4, tensor.size(0))):
#                     wandb_images[f"{tag}_{i}"] = self.wandb.Image(tensor[i])
#         self.wandb.log(wandb_images, step=step)