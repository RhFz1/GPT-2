import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import time
import math
import inspect
from dataclasses import dataclass


# Configuration for the GPT model, including hyperparameters
@dataclass
class ModelConfig:
    block_size: int = 1024       # Context window size
    vocab_size: int = 50257     # Number of tokens in the vocabulary
    n_layer: int = 12           # Number of transformer layers
    n_head: int = 12            # Number of attention heads
    n_embd: int = 768           # Embedding dimension
    attn_pdrop: float = 0.3     # Dropout rate for attention


# Multi-layer perceptron (MLP) used in the transformer blocks
class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)   # Expand embedding dimension
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd) # Project back to original size
        self.c_proj.NANOGPT_SCALE_INIT = 1                        # Custom initialization scale
        self.act = F.gelu                                          # Activation function
        self.dropout = nn.Dropout(config.attn_pdrop)              # Dropout for regularization
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.c_fc(x))                                # Forward pass through first layer
        h2 = self.c_proj(h)                                       # Forward pass through projection layer
        return self.dropout(h2)                                   # Apply dropout


# Causal Self-Attention Mechanism
class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head               # Size of each attention head
        assert self.head_dim * self.n_head == self.n_embd, 'n_embd should be divisible by n_head'
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)    # Query, key, and value projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)        # Output projection
        self.c_proj.NANOGPT_SCALE_INIT = 1                       # Custom initialization
        self.scale = 1 / (self.head_dim ** 0.5)                  # Scale for attention scores
        self.dropout = nn.Dropout(config.attn_pdrop)            # Dropout for attention weights
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))  # Causal mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)                                    # Compute query, key, value
        q, k, v = qkv.split(self.n_embd, dim=2)                 # Split into separate tensors
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Scaled dot-product attention with causal masking
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)        # Reassemble outputs
        y = self.c_proj(y)                                      # Project outputs back
        return y


# Transformer block containing attention and MLP layers
class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)                # Layer normalization
        self.attn = CausalSelfAttention(config)                # Attention layer
        self.ln_2 = nn.LayerNorm(config.n_embd)                # Layer normalization
        self.mlp = MLP(config)                                 # Feedforward MLP
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))                        # Residual connection for attention
        x = x + self.mlp(self.ln_2(x))                         # Residual connection for MLP
        return x


# GPT model definition
class GPT(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embedding
                wpe=nn.Embedding(config.block_size, config.n_embd),  # Positional embedding
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer blocks
                ln_f=nn.LayerNorm(config.n_embd)                     # Final normalization
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Output projection

        self.transformer.wte.weight = self.lm_head.weight  # Weight tying
        self.apply(self._init_weights)                     # Apply custom initialization
    
    def _init_weights(self, module):
        std = 0.02  # Standard deviation for initialization
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = (2 * ModelConfig.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.transformer.wpe.weight.size(0), 'Input sequence length exceeds context size'
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)                  # Positional embeddings
        token_emb = self.transformer.wte(idx)                # Token embeddings
        x = token_emb + pos_emb                              # Combine embeddings

        for block in self.transformer.h:
            x = block(x)                                     # Pass through transformer blocks
        
        x = self.transformer.ln_f(x)                        # Final layer normalization
        logits = self.lm_head(x)                            # Compute logits

        loss = None
        if targets is not None:
            # Compute cross-entropy loss for language modeling
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

class DataLoaderLite():

    def __init__(self, B, T, process_rank, num_processes):
        self.pos = 0
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.enc = tiktoken.get_encoding('gpt2')
        text = open('assets/input.txt', 'r').read()
        tokens = self.enc.encode(text)
        self.tokens = torch.tensor(tokens)
        if master_process:
            print(f"loaded {len(self.tokens)} tokens")
        self.pos = self.B * self.T * self.process_rank
    def get_next_batch(self):
        B, T = self.B, self.T
        buff = self.tokens[self.pos: self.pos + B*T + 1]
        x = buff[:-1].view(B, T)
        y = buff[1:].view(B, T)
        self.pos += B * T * self.num_processes
        if self.pos + (B * T * self.num_processes + 1)  > len(self.tokens):
            self.pos = self.B * self.T * self.process_rank
        return x, y


max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 50
def get_lr(itr):

    # Linear warmup for warmup_steps
    if itr < warmup_steps:
        return max_lr * (itr + 1) / warmup_steps
    # Min learning for greater than max_steps
    if itr > max_steps:
        return min_lr
    
    # Cosine annealing
    decay_ratio = (itr - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + (max_lr - min_lr) * coeff

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1 # this flag sets ddp run.

if ddp:
    # for ddp it is advised to use CUDA, haven't tried for CPU configs
    assert torch.cuda.is_available(), "Need CUDA for DDP maybe."
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    print(f'Using device: {device}')

ac_device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT(ModelConfig(vocab_size=50304))
model.eval()
model.to(device)
model = torch.compile(model)

total_batch_size = 524288 # 2**19, ~0.5M tokens
B,T = 16, 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, 'Batch size must be divisible by B * T'
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total batch size: {total_batch_size}")
    print(f"Grad Accum steps: {grad_accum_steps}")

trainloader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size)

torch.set_float32_matmul_precision('high')

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model =  model.module if ddp else model

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = trainloader.get_next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=ac_device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # in ms
    tokens_per_sec = (grad_accum_steps * trainloader.B * trainloader.T * ddp_world_size) / (t1 - t0)
    if master_process:
        print(f"Step: {step + 1}| Loss: {loss_accum.item():.6f}| Lr: {lr:.4e}| Norm: {norm:.4f}| dt: {dt:.2f}ms| tok/sec: {tokens_per_sec:.2f}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
if ddp:
    destroy_process_group()
import sys; sys.exit()
enc = tiktoken.get_encoding('gpt2')

max_len = 128
num_sequences = 1
x = enc.encode("RICHARD:A deadly groan, like life and death's departing.")
x = torch.tensor(x, device=device).unsqueeze(0)

while x.size(1) < max_len:
    
    logits, _ = model(x)

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

torch.save({
    'model_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict()
}, 'model.pt')


if ddp:
    destroy_process_group()