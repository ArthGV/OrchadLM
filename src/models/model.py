import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
 
from src.models.base_model import BaseLM
from src.utils.config import Config

class TransformerLM(BaseLM):
    def __init__(self, vocab_size: int, model_config: Config):
        super().__init__(model_config)
        self.vocab_size = vocab_size
        self.embed      = nn.Embedding(vocab_size, model_config.embed_dim)
        self.pos_embed  = nn.Embedding(model_config.context_len, model_config.embed_dim)
        self.blocks     = nn.Sequential(*[
            TransformerBlock(model_config.embed_dim, model_config.n_heads)
            for _ in range(model_config.n_layers)
        ])
        self.norm = nn.LayerNorm(model_config.embed_dim)   # final norm before head
        self.head = nn.Linear(model_config.embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T    = x.shape
        tok_emb = self.embed(x)                                      # (B, T, E)
        pos_emb = self.pos_embed(torch.arange(T, device=x.device))  # (T, E)
        
        x = tok_emb + pos_emb   # (B, T, E)
        x = self.blocks(x)      # (B, T, E)
        x = self.norm(x)        # final norm
        x = self.head(x)        # (B, T, V)
        return x

    def save_path(self) -> str:
        return self.name + f'_cl{self.model_config.context_len}_ed{self.model_config.embed_dim}_nh{self.model_config.n_heads}_nl{self.model_config.n_layers}'

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, n_heads)
        self.ff   = self.net = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim),
                        nn.ReLU(),
                        nn.Linear(embed_dim, embed_dim),
                    )
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # pre-norm + residual
        x = x + self.ff(self.ln2(x))     # pre-norm + residual
        return x
    
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, E)

        # 1. project to Q, K, V
        q = self.q_proj(x)   # (B, T, E)
        k = self.k_proj(x)   # (B, T, E)
        v = self.v_proj(x)   # (B, T, E)

        # 2. split into heads
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)   # (B, H, T, D)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)   # (B, H, T, D)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)   # (B, H, T, D)

        # 3. scaled dot-product attention scores
        scores = einsum(q, k, 'b h t d, b h s d -> b h t s') * self.scale   # (B, H, T, T)

        # 4. causal mask — prevent attending to future tokens
        T    = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # 5. softmax over key dimension
        attn = F.softmax(scores, dim=-1)   # (B, H, T, T)

        # 6. weighted sum of values
        out = einsum(attn, v, 'b h t s, b h s d -> b h t d')   # (B, H, T, D)

        # 7. merge heads back
        out = rearrange(out, 'b h t d -> b t (h d)')   # (B, T, E)

        return self.o_proj(out)