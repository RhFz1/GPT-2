import torch 
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

