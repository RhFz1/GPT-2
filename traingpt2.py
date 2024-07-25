import os
import glob
import torch
import time
import math 
import torch.utils
import torch.distributed as dist
from argparse import ArgumentParser
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from dataloader import DataLoader
from config import GPTConfig
from model import GPT
from dotenv import load_dotenv

load_dotenv()

parser = ArgumentParser(description="Train Run Counter")
parser.add_argument("--run_num", type=int, required=True, help="Enter the training run, set to zero In case of a fresh run, else verify from models.")
args = parser.parse_args()

# Paths
train_log_path = os.path.join(os.getenv('LOG_PATH'), 'trainlogs.txt')
val_log_path = os.path.join(os.getenv('LOG_PATH'), 'vallogs.txt')
model_path = glob.glob(os.getenv('REGISTRY_PATH')+'/' + '*.pt')
model_path = model_path[-1] if len(model_path) > 0 else None
model_out_path = os.environ.get('REGISTRY_PATH')

# Setting up DDP Run
ddp = int(os.environ.get('RANK', -1)) != -1

# this block configures the ddp settings, adding an else to it to perform vanilla run if ddp is not enabled.
if ddp:
    # i have read the torch docs. I think we can use the gloo backend to run ddp on a multi core cpu
    assert dist.is_gloo_available(), "Gloo backend is not available for DDP."
    init_process_group(backend='gloo')
    ddp_rank = int(os.environ.get('RANK'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # auto detect device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    master_process = True
    print(f"Running on {device}.")

# Setting up the train params
# Dataloder params
total_batch_size = 262144 # this is the total batch size virtually before optimizer step.
B = 32 # micro batch size
T = 256 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
# Learning rate params
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 50
max_steps = 10000
gain_for_next = 1.25
# Evaluation Params
eval_iters = 5
eval_interval = 20
save_step = 2 # save the model once evaluation is done.
test_step = 50
# Torch precision settings
torch.set_float32_matmul_precision('high')
# Setting the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Rule of thumb

# Use the 'NCCL' backend for distributed GPU training.
# Use the 'Gloo' backend for distributed CPU training.

# Loading if existing trained model present
checkpoint = None # using this for setting learning rate.
best_loss = float('inf')

if model_path:
    print(f"Loading the model: {model_path}......")
    checkpoint = torch.load(model_path)

    # new_keys = {key: key[10:] for key in checkpoint['model'].keys()}
    # checkpoint['model'] = OrderedDict((new_keys.get(k, k), v) for k, v in checkpoint['model'].items())

    model_args = checkpoint['model_args']
    model = GPT(model_args)
    model.load_state_dict(checkpoint['model'])

    print(f"Previous model loaded!, best loss: {checkpoint['best_loss']:.4f}")

    # Reinit lr
    max_lr = checkpoint['lr'] * gain_for_next
    min_lr = max_lr * 0.1
    best_loss = checkpoint['best_loss']
    
else:
    model_args = GPTConfig(vocab_size=50304)
    model = GPT(model_args)
    print(f"No previous model found making new model.")

# DDP settings
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
raw_model = model.module if ddp else model

# Placing the model on the GPU
model = model.to(device)
# It fuses the models individual operations wherever possible, making it faster while training.
model = torch.compile(model)
# Setting up the optimizer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=checkpoint['lr'] * gain_for_next if checkpoint else 6e-4, device_type=device)

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


# logging if master process
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# Setting up the dataloader
train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# loss tracking dictionary
losses = {'val_loss': [], 'train_loss': []}

# logging the train run
with open(train_log_path, 'a') as file:
    file.write(f"Training run: {args.run_num}\n")
file.close()
with open(val_log_path, 'a') as file:
    file.write(f"Training run: {args.run_num}\n")

# Starting the train run
for i in range(max_steps):
    
    # Start time
    t0 = time.time()

    # Testing the model to see how it performs, mid training.
    if i != 0 and i % test_step == 0:
        model.eval()
        prompt = "I'm feeling dizzy"
        print(model._generate(prompt, max_len=200, temperature=1.0))

    val_loss_accum = 0.0
    # Evaluation of the model, validation.
    if i != 0 and i % eval_interval == 0:
        model.eval()
        
        with torch.no_grad():
            val_loader.reset()
            for _ in range(eval_iters):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                loss = loss / eval_iters
                val_loss_accum += loss.detach()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = val_loader.B * val_loader.T * eval_iters / dt
        with open(val_log_path, 'a') as f:
            f.write(f"step {i:4d} | lr: {lr:.5f} | val_loss: {val_loss_accum:.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}\n")
        print(f"step {i:4d} | lr: {lr:.5f} | val_loss: {val_loss_accum:.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    # Tracking val loss
    losses['val_loss'].append(val_loss_accum)

    # do one step of grad accum iters
    model.train()
    loss_accum = 0.0
    optimizer.zero_grad()
    for microstep in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # torch.autocast is used to perform mixed precision training.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)  
        # Normalizing the loss, mathematical correctness for grad. accumulation.
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() 
        loss.backward()

        if ddp:
            model.require_backward_grad_sync = (microstep == grad_accum_steps - 1)
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    losses['train_loss'].append(loss_accum)

    if best_loss > losses['train_loss'][-1]:
        best_loss = losses['train_loss'][-1] 
        if i > 0 and i % save_step == 0:
            print(f"Preparing to save model with best loss: {losses['train_loss'][-1]:.4f}")
            checkpoint = {
                'model' : raw_model.state_dict(),
                'config' : raw_model.config,  
                'model_args' : model_args,
                'step' : i,
                'best_loss': best_loss,
                'lr': lr # this will throw an error when save step set to 1. 
            }
            print(f"saving model checkpoint to {model_out_path}, with best_loss: {best_loss:.4f}")
            torch.save(checkpoint, os.path.join(model_out_path, f'ckpt_{args.run_num}.pt'))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate
    lr = get_lr(i)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    
    t2 = time.time()
    # time delta in s
    dt = (t2 - t0) 
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t2 - t0)
    print(f"step {i:4d} | lr: {lr:.5f} | loss: {loss_accum.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    # writing to log file to track the changes.
    with open(train_log_path, 'a') as f:
        f.write(f"step {i:4d} | lr: {lr:.5f} | loss: {loss_accum.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}\n")