import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import time
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
        self.c_proj.NANOGPT_SCALE_INIT = 1
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
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.scale = 1 / (self.head_dim ** 0.5)
        self.dropout = nn.Dropout(config.attn_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor)->torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
        
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

        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = (2 * ModelConfig.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean = 0.0, std=std)
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None)->torch.Tensor:
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

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
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


class DataLoaderLite():

    def __init__(self, B, T):
        self.pos = 0
        self.B = B
        self.T = T
        self.enc = tiktoken.get_encoding('gpt2')
        text = open('assets/input.txt', 'r').read()
        tokens = self.enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

    def get_next_batch(self):
        B, T = self.B, self.T
        buff = self.tokens[self.pos : self.pos + B*T + 1]
        x = buff[:-1].view(B, T)
        y = buff[1:].view(B, T)
        self.pos += B * T
        if self.pos + B * T + 1  > len(self.tokens):
            self.pos = 0
        return x, y

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

print(f'Using device: {device}')

model = GPT(ModelConfig())
model.eval()
model.to(device)

B,T = 8, 1024
trainloader = DataLoaderLite(B, T)

torch.set_float32_matmul_precision('high')

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.99, 0.999))

for i in range(50):
    t0 = time.time()
    x, y = trainloader.get_next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # in ms
    tokens_per_sec = (trainloader.B * trainloader.T) / (t1 - t0)
    print(f"Step: {i + 1}, Loss: {loss.item():.6f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

max_len = 32
num_sequences = B

while x.size(1) < max_len:
    
    logits = model(x)

    logits = logits[:, -1, :] # (B, vocab_size)

    probs = F.softmax(logits, dim = 1)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    # select a token from the top-k probabilities
    # note: multinomial does not demand the input to sum to 1
    ix = torch.multinomial(topk_probs, 1) # (B, 1)
    # gather the corresponding indices
    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
    # append to the sequence
    x = torch.cat((x, xcol), dim=1)


for row in range(num_sequences):
    tokens = x[row, : max_len].tolist()
    print(enc.decode(tokens))