import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.visualization.cka import collect_cka, linear_cka

torch.manual_seed(1337)
x = torch.randn(32, 8)
assert torch.allclose(linear_cka(x, x), torch.tensor(1.0), atol=1e-5)
config = GPTConfig(block_size=5, vocab_size=32, n_layer=3, n_head=4, n_embd=16)
model = GPT(config).eval()


def batch_fn(_split):
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size))
    return input_ids, input_ids


matrix = collect_cka(model, batch_fn, batches=2, max_positions=16)
assert matrix.shape == (config.n_layer, config.n_layer)
assert torch.isfinite(matrix).all()
assert torch.allclose(matrix, matrix.T, atol=1e-5)
print("lens CKA check passed")
