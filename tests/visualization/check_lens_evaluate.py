import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.visualization.evaluate import evaluate_tuned_lens
from src.visualization.tuned_lens import TunedLens

torch.manual_seed(1337)
config = GPTConfig(block_size=5, vocab_size=32, n_layer=3, n_head=4, n_embd=16)
model = GPT(config).eval()
lens = TunedLens(config.n_layer, config.n_embd).eval()


def batch_fn(_split):
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    return x, torch.roll(x, shifts=-1, dims=1)


metrics = evaluate_tuned_lens(model, lens, batch_fn, batches=2)
assert all(value.shape == (config.n_layer, 2 * 2 * config.block_size) for value in metrics.values.values())
assert all(value.shape == (config.n_layer,) for value in metrics.means.values())
assert all(torch.isfinite(value).all() for value in metrics.means.values())
print("lens evaluation check passed")
