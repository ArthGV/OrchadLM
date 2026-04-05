import torch
from torch.utils.data import DataLoader
from datetime import datetime

class Trainer:
    def __init__(self, device, model, optimizer, loader: DataLoader):
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

    def train(self, 
              epochs: int, 
              print_gap: int|None = None,
              save_gap: int|None = None,
              save: bool = False):
        starting_time = datetime.now()
        for epoch in range(epochs):
            loss = self.train_epoch()
            if print_gap:
                if (epoch % print_gap == 0) or (epoch == epochs-1):
                    elapsed = datetime.now() - starting_time
                    total_seconds = int(elapsed.total_seconds())
                    h, remainder  = divmod(total_seconds, 3600)
                    m, s          = divmod(remainder, 60)
                    print(f"Epoch {epoch+1} - Loss {loss:.4f} - Time {h:02d}h {m:02d}m {s:02d}s")
            if save and save_gap:
                pass
                #self.save()
        if save:
            pass
            #self.save()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))