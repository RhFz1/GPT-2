import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils
from config import GPTConfig

master_process = True
# Sampling from shakespere
class DataLoader():
    def __init__(self, B, T, process_rank, num_processes, split):

        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        data_path = os.getenv('DATA_PATH') # Getting the datastore path
        shards = os.listdir(data_path) # Listing the files in the datastore path
        shards = [s for s in shards if split in s] # Filtering the files based on the split
        shards = sorted(shards) # Sorting the files 
        shards = [os.path.join(data_path, s) for s in shards] # Creating the full path of the files
        
        self.shards = shards

        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32) # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt
    
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y