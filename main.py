import os
import torch
import torch.nn.functional as F
import tiktoken
import time
import math

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
warmup_steps = 100
max_steps = 400
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

torch.save({
    'model_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict()
}, './artifacts/model_n.pt')

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
}, '/artifacts/model.pt')


if ddp:
    destroy_process_group()