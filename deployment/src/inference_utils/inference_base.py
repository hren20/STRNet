# inference_base.py

from abc import ABC, abstractmethod
from typing import List, Dict
import torch
import numpy as np

class BaseInferenceTrainer(ABC):
    def __init__(self, config: Dict, checkpoint_path: str):
        self.config = config
        self.device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def init_inference_action_stats(self):
        self.ACTION_STATS = {}
        self.ACTION_STATS['min'] = np.array([-2.5, -4])
        self.ACTION_STATS['max'] = np.array([5, 4])

    @abstractmethod
    def prepare_inputs(self, image_paths: List[str]) -> torch.Tensor:
        """将图像路径转换为模型输入张量"""
        pass

    @abstractmethod
    def predict_actions(self, obs_images: torch.Tensor, **kwargs) -> Dict:
        """根据输入图像预测动作"""
        pass
