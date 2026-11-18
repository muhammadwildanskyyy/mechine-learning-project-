"""Deep learning pipelines and model registry for image classification."""

from ml_project.deep_learning.config import EvalConfig, TrainConfig
from ml_project.deep_learning.evaluator import run_evaluation
from ml_project.deep_learning.models import create_model, get_architecture_spec, list_architectures
from ml_project.deep_learning.trainer import run_training

__all__ = [
    "EvalConfig",
    "TrainConfig",
    "create_model",
    "get_architecture_spec",
    "list_architectures",
    "run_evaluation",
    "run_training",
]
