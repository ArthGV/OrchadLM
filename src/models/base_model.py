import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    
class Trainer:
    def __init__(self, device, model: BaseLM, optimizer, loader: DataLoader):
        self.device = device
        self.model     = model
        self.optimizer = optimizer
        self.loader    = loader

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y in self.loader:
            x, y   = x.to(self.device), y.to(self.device)
            _, loss = self.model.train_step(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)

    def train(self, steps: int):
        for step in range(steps + 1):
            loss = self.train_epoch()
            if step % 500 == 0:
                print(f"step {step:5d}  loss {loss:.4f}")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))