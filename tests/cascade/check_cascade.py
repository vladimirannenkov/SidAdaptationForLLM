"""Fast CPU checks for Cascade-SID loss isolation and chunk equivalence."""
import os
import sys
import tempfile
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.cascade.checkpoint import load_checkpoint, save_checkpoint
from src.cascade.losses import cascade_three_zone_loss
from src.model.gpt import GPT, GPTConfig
from src.sid.forward import embed, forward_range

torch.manual_seed(0)
model, cfg = GPT(GPTConfig()), GPTConfig()
x = torch.randint(cfg.vocab_size, (2, cfg.block_size))
y = torch.randint(cfg.vocab_size, (2, cfg.block_size))
h0 = forward_range(model, embed(model, x), 0, 6).detach()
u = torch.randn_like(h0, requires_grad=True)
alpha = torch.tensor(.01, requires_grad=True)
loss_a, _ = cascade_three_zone_loss(model, h0, [], [], u, alpha, y, chunk_size=64)
loss_b, _ = cascade_three_zone_loss(model, h0, [], [], u, alpha, y, chunk_size=cfg.block_size)
assert torch.allclose(loss_a, loss_b, atol=1e-5), (loss_a, loss_b)
loss_a.backward()
assert u.grad is not None and alpha.grad is not None
assert h0.grad is None

alphas = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(.01)) for _ in range(6)])
optimizer = torch.optim.AdamW(list(model.transformer.h[6].parameters()) + [alphas[0]])
with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "latest.pt")
    save_checkpoint(path, model, alphas, optimizer, stage=0, stage_step=3,
                    tokens_processed=123, config={"run_id": "test"})
    state = load_checkpoint(path)
    assert state["stage"] == 0 and state["stage_step"] == 3
    assert state["tokens_processed"] == 123
    assert torch.allclose(state["alphas"], torch.full((6,), .01))

print("Cascade loss, isolation, and checkpoint resume format passed.")
