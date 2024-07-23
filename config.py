from dataclasses import dataclass

@dataclass
class GPTConfig:
    layers: int = 12
    vocab_size: int = 50257
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 1024
    dropout: float = 0.3
 