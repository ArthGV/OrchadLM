import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, context_len: int):
        self.tokens      = tokens
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, i):
        x = self.tokens[i     : i + self.context_len]
        y = self.tokens[i + 1 : i + self.context_len + 1]
        return x, y