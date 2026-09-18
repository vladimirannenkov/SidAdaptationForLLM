import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.visualization.checkpoint import load_e2e_checkpoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
checkpoint_path = os.path.join(REPO_ROOT, "checkpoints", "checkpoint_100pct.pt")
model, checkpoint = load_e2e_checkpoint(checkpoint_path)

assert checkpoint["optimizer_step"] == 1524
assert checkpoint["tokens_processed"] == 99_876_864
assert model.config.n_layer == 12
assert model.config.n_embd == 272
assert all(not parameter.requires_grad for parameter in model.parameters())

with torch.no_grad():
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    logits, loss = model(input_ids, input_ids.clone())
assert logits.shape == (1, 8, model.config.vocab_size)
assert loss is not None and torch.isfinite(loss)
print("lens checkpoint check passed")
