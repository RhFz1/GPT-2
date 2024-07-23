import torch
import tiktoken
import torch.nn.functional as F
import torch.utils
from config import GPTConfig

# Sampling from shakespere
class DataLoader():
    def __init__(self, B, T, process_rank, num_processes):

        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open('input.txt', 'r') as file:
            text = file.read()

        tokenizer = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(tokenizer.encode(text))
        if master_process:
            print(f"Total number of tokens: {len(self.tokens)}")
        self.current_pos = self.B * self.T * self.process_rank
    
    def reset(self):
        self.current_pos = torch.randint(len(self.tokens) - self.B * self.T, (1,)).item()

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos:self.current_pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_pos += B * T * self.num_processes

        if self.current_pos + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_pos = 0
        
        return x.to(GPTConfig.device), y.to(GPTConfig.device)