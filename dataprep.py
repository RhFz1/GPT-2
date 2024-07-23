import os
import tiktoken
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


local_dir = 'data'
remote_name = "medqca"
shard_size = int(1e6)

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
