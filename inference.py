import torch
import os
from traingpt2 import GPT, GPTConfig

model_path = '/home/syednoor/Desktop/FAIR/GPT-2/models'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT(GPTConfig(vocab_size=50304))

model = model.to(device)
checkpoint = torch.load(os.path.join(model_path, 'ckpt_00.pt'), map_location=device)

model.load_state_dict(checkpoint['model'])

prompt = " "

print(model._generate(prompt, max_len=100, temperature=0.2))