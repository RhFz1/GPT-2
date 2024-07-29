import torch
import os
from model import GPT
from config import GPTConfig
from collections import OrderedDict

model_path = '/home/syednoor/Desktop/FAIR/GPT-2/models'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT(GPTConfig(vocab_size=50304))

model = model.to(device)
checkpoint = torch.load(os.path.join(model_path, 'ckpt_04.pt'), map_location=device)
new_keys = {key: key.replace('_orig_mod.', '') for key in checkpoint['model'].keys()}
checkpoint['model'] = OrderedDict((new_keys.get(k, k), v) for k, v in checkpoint['model'].items())

model.load_state_dict(checkpoint['model'])

prompt = " "

print(model._generate(prompt, max_len=100, temperature=0.2))