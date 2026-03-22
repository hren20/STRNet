from .inference_base import BaseInferenceTrainer
from .inference_strnetnew import InferenceSTRNetNewTrainer

__all__ = ['BaseInferenceTrainer', 'InferenceSTRNetNewTrainer']

MODEL_REGISTRY = {
    "strnetnew": InferenceSTRNetNewTrainer,
}