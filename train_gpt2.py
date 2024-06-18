import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional



@dataclass
class GPTConfig:
    layers: int = 12
    vocab_size: int = 50257
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 1024

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
        # setting it to bias only for naming reasons.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size)) # (1, 1, block_size, block_size)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (B, T, 3 * n_embd) -> (B, T, 3, n_head, head_dim) -> (B, 3, T, n_head, head_dim)


        qkv = self.c_attn(x).reshape(B, T, 3, self.n_head, self.n_embd // self.n_head).permute(0, 2, 1, 3, 4)

        q, k, v = qkv.reshape(3, B * self.n_head, T, self.n_embd // self.n_head).unbind(dim=0) # q, k, v are (B * n_head, T, head_dim)
    
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B * n_head, T, head_dim) x (B * n_head, head_dim, T) -> (B * n_head, T, T)
        attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        y = attn @ v # (B * n_head, T, T) x (B * n_head, T, head_dim) -> (B * n_head, T, head_dim)
        # y = y.permute(0, 2, 1, 3).contiguous().view(B, T, self.n_embd) # (B, T, n_head, head_dim) -> (B, T, n_embd)
        y = y.view(B, self.n_head, T, self.n_embd // self.n_head).permute(0, 2, 1, 3).reshape(B, T, self.n_embd)
        out = self.c_proj(y) # (B, T, n_embd)
        
        return out

class MLP(nn.Module):
    # MLP after attention
    # Intuitively used for the model to learn about the attention output

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # (B, T, n_embd) -> (B, T, 4 * n_embd)
        self.act = nn.GELU(approximate='tanh') 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # (B, T, 4 * n_embd) -> (B, T, n_embd

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # Layer Normalization preferred over batch normalization, as it is more robust.
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x)) # Skip connection as per paper.
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self. lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x is (B, T) where B is batch size and T is block size
        B, T = x.shape
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(T, dtype=torch.long, device=x.device) # (T) as we cannot expect to a sequence of length greater than block size.
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(x) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)

        # This is the forward pass of the transformer model.
        # It is a recursive flow of input through the transformer blocks.
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        return logits
    
    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2") -> 'GPT':
        """Loads the GPT-2 model weights from the Hugging Face's transformers library."""
        print("Loading the GPT-2 model weights from Hugging Face's transformers library.")


        # Passing the config and instantiating the model.
        config = GPTConfig()
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # Init a hugging face model and load the weights.
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

model = GPT.from_pretrained("gpt2")
print("loaded!!")