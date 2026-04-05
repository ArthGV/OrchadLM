import torch
import torch.nn as nn
import torch.nn.functional as F
 
from src.models.base_model import BaseLM
from src.utils.config import Config

class TransformerLM(BaseLM):
    def __init__(self, 
                 vocab_size: int, 
                 model_config: Config):
        super().__init__(model_config)
        self.embed    = nn.Embedding(vocab_size, model_config.embed_dim)
        self.pos_embed = nn.Embedding(model_config.context_len, model_config.embed_dim)
        self.blocks   = nn.Sequential(*[TransformerBlock(model_config.embed_dim, model_config.n_heads) for _ in range(model_config.n_layers)])
        self.head     = nn.Linear(model_config.embed_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        B, T    = x.shape
        tok_emb = self.embed(x)                                  # (B, T, E)
        pos_emb = self.pos_embed(torch.arange(T, device=x.device))  # (T, E)
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        logits  = self.head(x)                                   # (B, T, V)
        return logits

    def save_path(self) -> str:
        return self.name + f'_cl{self.model_config.context_len}_ed{self.model_config.embed_dim}_nh{self.model_config.n_heads}_nl{self.model_config.n_layers}'

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim)

    def forward(self, x):
        T = x.size(1)
        # Causal mask: each position can only attend to past positions
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)           # residual + norm
        x = self.ln2(x + self.ff(x))         # residual + norm
        return x