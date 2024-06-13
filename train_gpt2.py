import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional



@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, as single concatenated matrix
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # heads and head size
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size)) # (1, 1, block_size, block_size)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (B, T, 3 * n_embd) -> (B, T, 3, n_head, head_dim) -> (B, 3, T, n_head, head_dim)
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, self.n_embd // self.n_head).permute(0, 2, 1, 3, 4)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T) where nh is number of heads, hs is head size
        q, k, v = qkv.unbind(dim=1) # q, k, v are (B, T, n_head, head_dim)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        