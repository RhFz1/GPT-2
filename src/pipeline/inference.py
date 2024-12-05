import torch
import torch.nn.functional as F
import tiktoken
from src.constants.model_arch import GPT, ModelConfig

checkpoint = torch.load('./artifacts/model_n.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict_model = checkpoint['model_dict']
state_dict_optim = checkpoint['optimizer_dict']

state_dict_model = {k.replace('_orig_mod.', ''):v for k, v in state_dict_model.items()}

model = GPT(ModelConfig(vocab_size=50304))
model.load_state_dict(state_dict_model)
model.to(device=device)

model.eval()
with torch.no_grad():
    enc = tiktoken.get_encoding('gpt2')

    max_len = 324
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