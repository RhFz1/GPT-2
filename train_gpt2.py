import torch
import time
import tiktoken
import inspect
import math 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
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
    device: str = 'cpu' if not torch.cuda.is_available() else 'cuda'

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, as single concatenated matrix
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT2SCALEINIT = 1
        
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
    
        # attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B * n_head, T, head_dim) x (B * n_head, head_dim, T) -> (B * n_head, T, T)
        # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v # (B * n_head, T, T) x (B * n_head, T, head_dim) -> (B * n_head, T, head_dim)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # (B, T, 4 * n_embd) -> (B, T, n_embd)
        self.c_proj.GPT2SCALEINIT = 1

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


class Tokenizer():
    def __init__(self, enc_type: str = "gpt2") -> None:
        self.encoder = tiktoken.get_encoding(enc_type)
    def encode(self, text: str) -> torch.Tensor:

        tokens = self.encoder.encode(text) # Now the text is converted to tokens gpt2 has roughly 50k tokens (vocab_size). each token value corresponding to a word segment.
        assert len(tokens) <= GPTConfig.block_size, "Token length exceeds the block size of the model." # The model has a block size of 1024 tokens.
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) # (1, T) this is done for adding a batch dimension.
        tokens = tokens.to(GPTConfig.device) # moving the tokens to the GPU.
        return tokens
    def decode(self, tokens: torch.Tensor) -> str:

        B, T = tokens.shape

        assert B == 1, "Only supports batch size of 1 for now." # Only supporting batch size of 1 for now, will add support for multiple batch sizes later.
        text = self.encoder.decode(tokens.squeeze().tolist())

        return text 

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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing between the token embeddings and the final linear layer, also known as parameter tying.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    def _init_weights(self, module) -> None:
        # Here in this function we try to initialize the weights of the model.
        # As per the paper we are supposed to rescale the weights on the basis of number of skip connections.
        # We n layers and each layer has 2 skip connections (one after attention and one after MLP).
        # For proof of reasoning and math refer to the paper.

        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT2SCALEINIT'):
                std *= (2 * self.config.layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
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
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1)) # (B * T, vocab_size), (B * T). Cross Entropy accepts input in this format (refer PyTorch documentation).
        
        return logits, loss
    
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
    
    def _generate(self, prompt: str, max_len: int = 20, temperature: float = 1.0) -> str:
        self.eval() # Setting to eval mode, for efficiency.
        tokenizer = Tokenizer() # Tokenizer class is used to encode the prompt.
        x = tokenizer.encode(prompt) # (1, T)

        for _ in range(max_len):
            logits, _ = self.forward(x)
            logits = logits[:, -1, :] / temperature # (B, vocab_size) focusing on the last token, for prediction.

            probs = F.softmax(logits, dim=-1) # (B, vocab_size) applying softmax to get probabilities.
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1) sampling the next token from the distribution.
            x = torch.cat((x, next_token), dim=-1) # (1, T + 1) appending the next token to the sequence.\
        
        text = tokenizer.decode(x) # Decoding the tokens to text.
        return text
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):

        # segregating the parameters of the model into two groups, one with weight decay and one without.
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # all the parameters with 2D dim will be decayed, rest others shall not which include biases and layernorm weights.
        decay_parameters = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_parameters = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": non_decay_parameters, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_parameters)
        num_non_decay_params = sum(p.numel() for p in non_decay_parameters)

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# Sampling from shakespere
class DataLoader():
    def __init__(self, B, T):

        self.B = B
        self.T = T

        with open('input.txt', 'r') as file:
            text = file.read()

        tokenizer = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(tokenizer.encode(text))

        print(f"Loaded text of length {len(text)} and {len(self.tokens)} tokens.")
        print(f"Epoch size: {len(self.tokens) // (B * T)} batches.")
        self.current_pos = 0
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos:self.current_pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_pos += B * T

        if self.current_pos + B * T + 1 > len(self.tokens):
            self.current_pos = 0
        
        return x, y
    
device = GPTConfig.device

model = GPT(GPTConfig(vocab_size=50304))
model = torch.compile(model)

total_batch_size = 524288
B = 16
T = 128
assert total_batch_size % (B * T) == 0, "Batch size is not a multiple of the total batch size."
grad_acc_steps = total_batch_size // (B * T)
print(f"Total batch size: {total_batch_size}")
print(f"Gradient accumulation steps: {grad_acc_steps}")

train_loader = DataLoader(B = B, T = T)

torch.set_float32_matmul_precision('high')
# run train loop

model.train()

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup iters
    if it < warmup_steps:
        return min_lr + (max_lr - min_lr) * it / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay as per paper.
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0  + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

for i in range(max_steps):
    t0 = time.time()
    loss_accum = 0.0
    optimizer.zero_grad()
    for microstep in range(grad_acc_steps):
        x, y = train_loader.next_batch()
        logits, loss = model(x, y)
        loss_accum += loss.detach() / grad_acc_steps
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate
    lr = get_lr(i)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    t1 = time.time()
    # time delta in s
    dt = (t1 - t0) 
    tokens_per_sec = (train_loader.B * train_loader.T * grad_acc_steps) / (t1 - t0)
    print(f"step {i:4d} | lr: {lr:.5f} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0)

# Eval the model

model.eval()
prompt = "The king was riding his horse"

print(model._generate(prompt, max_len=100, temperature=1.0))