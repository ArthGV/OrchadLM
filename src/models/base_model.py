import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from src.utils.config import Config

class BaseLM(nn.Module, ABC):
    def __init__(self, model_config: Config):
        super().__init__()
        self.model_config = model_config

    @property
    def meta(self) -> Config:
        return self.model_config.meta

    @property
    def name(self) -> str:
        return f'{self.model_config.meta.type}_{self.model_config.meta.model}_v{self.model_config.meta.version}'

    @abstractmethod
    def forward(self, x: torch.tensor):
        """Given token ids (B, T), return logits (B, T, V)."""
        ...
    
    @abstractmethod
    def save_path(self) -> str:
        """Given hyparameters, build the save path."""
        ...
    
    def compute_training_loss(self, ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(prediction.view(-1, prediction.size(-1)), ground_truth.view(-1))

    def train_step(self, x: torch.tensor, y: torch.tensor):
        logits = self.forward(x)
        loss = self.compute_training_loss(y, logits)
        return logits, loss