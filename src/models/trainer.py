import os
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
              save: bool = True):
        starting_time = datetime.now()
        out_folder_path = 'models/' + self.model.name + '/'
        os.makedirs(out_folder_path, exist_ok=True)
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
                self.save(out_folder_path + self.model.save_path() + f'_{epoch+1}.pth')
        if save:
            pass
            self.save(out_folder_path + self.model.save_path() + f'_{epoch+1}.pth')

    def save(self, path: str):
        self.model.save(path)
        print(f'Saved Model at: {path}')