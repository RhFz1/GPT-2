import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 384


class GPT(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                lnf = nn.LayerNorm(config.n_embd)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        B, T = x.size()
        assert T <= self.transformer['wpe'].weight.size(0), 'Input sequence length is longer than the context size'

        pos_emb = self.transformer['wpe'](torch.arange(T)) # (T, n_embd)
        token_emb = self.transformer['wte'](x) # (B, T, n_embd)
        x = token_emb + pos_emb # (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd)

        for block in self.transformer['h']:
            x = block(x) # (B, T, n_embd)
        
        x = self.lm_head(x) # (B, T, n_embd) -> (B, T, vocab_size)
        return x