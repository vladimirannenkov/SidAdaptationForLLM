import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.visualization.model_adapter import forward_with_states
from src.visualization.tuned_lens import TunedLens

torch.manual_seed(1337)
model = GPT(GPTConfig()).eval()
input_ids = torch.randint(0, model.config.vocab_size, (2, 11))
lens = TunedLens(model.config.n_layer, model.config.n_embd).eval()

with torch.no_grad():
    trajectory = forward_with_states(model, input_ids)
    translated_logits = lens.logits(
        trajectory.hidden_states[-1], model.config.n_layer - 1,
        model.transformer.ln_f, model.lm_head,
    )

assert torch.equal(translated_logits, trajectory.final_logits)
assert sum(parameter.numel() for parameter in lens.parameters()) == (
    model.config.n_layer * (model.config.n_embd ** 2 + model.config.n_embd)
)
print("tuned lens check passed")
