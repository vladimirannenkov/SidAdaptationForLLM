import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.visualization.model_adapter import forward_with_states

torch.manual_seed(1337)
model = GPT(GPTConfig()).eval()
input_ids = torch.randint(0, model.config.vocab_size, (2, 17))

with torch.no_grad():
    trajectory = forward_with_states(model, input_ids)
    reference_logits, _ = model(input_ids, input_ids.clone())

assert len(trajectory.hidden_states) == model.config.n_layer
assert all(state.shape == (2, 17, model.config.n_embd) for state in trajectory.hidden_states)
assert trajectory.final_logits.shape == (2, 17, model.config.vocab_size)
assert torch.equal(trajectory.final_logits, reference_logits)
print("lens model adapter check passed")
