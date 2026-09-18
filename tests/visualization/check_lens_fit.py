import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.visualization.fit import fit_tuned_lens
from src.visualization.tuned_lens import TunedLens

torch.manual_seed(1337)
model = GPT(GPTConfig(block_size=8, vocab_size=32, n_layer=3, n_head=4, n_embd=16)).eval()
lens = TunedLens(model.config.n_layer, model.config.n_embd)
optimizer = torch.optim.AdamW(lens.parameters(), lr=1e-3)


def batch_fn(_split):
    return torch.randint(0, model.config.vocab_size, (2, model.config.block_size))


history = fit_tuned_lens(model, lens, batch_fn, optimizer, train_steps=2)
assert len(history) == 2
assert len(history[-1]["layer_kl"]) == model.config.n_layer
assert all(torch.isfinite(torch.tensor(item["loss"])) for item in history)
print("lens fit check passed")
