import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

class BaseLM(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.tensor):
        """Given token ids (B, T), return logits (B, T, V)."""
        ...
    
    def compute_training_loss(self, ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(prediction.view(-1, prediction.size(-1)), ground_truth.view(-1))

    def train_step(self, x: torch.tensor, y: torch.tensor):
        logits = self.forward(x)
        loss = self.compute_training_loss(y, logits)
        return logits, loss