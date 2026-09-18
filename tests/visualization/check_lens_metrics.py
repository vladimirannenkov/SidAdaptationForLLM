import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.visualization.metrics import cross_entropy, entropy, forward_kl, masked_mean

log_probs = torch.log(torch.tensor([[[0.25, 0.75], [0.5, 0.5]]]))
reference = torch.log(torch.tensor([[[0.5, 0.5], [0.25, 0.75]]]))
targets = torch.tensor([[1, -1]])

assert torch.allclose(entropy(log_probs), torch.tensor([[0.562335, 0.693147]]), atol=1e-5)
assert torch.allclose(cross_entropy(log_probs, targets), torch.tensor([[0.287682, float("nan")]]), equal_nan=True)
assert torch.all(forward_kl(log_probs, reference) >= 0)
assert torch.allclose(masked_mean(cross_entropy(log_probs, targets)), torch.tensor(0.287682), atol=1e-5)
print("lens metrics check passed")
