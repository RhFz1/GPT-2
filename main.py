import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    attn_pdrop: float = 0.3


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.attn_pdrop)
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        assert self.head_dim * self.n_head == self.n_embd, 'n_embd should be divisible by n_head'
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.scale = 1 / (self.head_dim ** 0.5)
        self.dropout = nn.Dropout(config.attn_pdrop)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(0, 2, 1, 3, 4)
        q, k, v = qkv.unbind(dim=1)
        
        q = q.reshape(B * self.n_head, T, self.head_dim) # (B, 3, T, n_head, head_dim) -> 3 * (B * n_head, T,head_dim)
        k = k.reshape(B * self.n_head, T, self.head_dim)
        v = v.reshape(B * self.n_head, T, self.head_dim)

        w = (q @ k.transpose(-2, -1)) * self.scale # (B * n_head, T, head_dim) @ (B * n_head, head_dim, T) -> (B * n_head, T, T)
        w.masked_fill_(torch.tril(torch.ones(T, T, device=w.device, dtype=torch.bool), 1), float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w) # (B * n_head, T, T)
        
        a = w @ v # (B * n_head, T, T) @ (B * n_head, T, head_dim) -> (B * n_head, T, head_dim)
        a.reshape(B, self.n_head, T, self.head_dim).permute(0, 2, 1, 3).reshape(B, T, self.n_embd)
        a = self.c_proj(a)
        
        return a


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx: torch.Tensor)->torch.Tensor:
        B, T = idx.size()
        assert T <= self.transformer.wpe.weight.size(0), 'Input sequence length is longer than the context size'
        pos = torch.arange(0, T, dtype=torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        token_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = token_emb + pos_emb # (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd)

        for block in self.transformer.h:
            x = block(x) # (B, T, n_embd)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, n_embd) -> (B, T, vocab_size)
        return logits
    
    @staticmethod
    def from_pretrained(model_type: str)->nn.Module:
        
        assert model_type == 'gpt2', 'Only GPT2 is supported'
        from transformers import GPT2LMHeadModel
        print(f"Loading weights of {model_type}")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024)
        }[model_type]

        config = ModelConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
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
    

if __name__ == '__main__':
    model = GPT.from_pretrained(model_type='gpt2')
    print('Model Loaded!!')