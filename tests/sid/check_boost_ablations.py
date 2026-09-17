"""Проверка новых абляций train_sid_p_boost.py (2026-09-21):
1. frozen_readout(model, x) даёт ТО ЖЕ значение, что и обычный readout(model, x)
   (те же веса, только detach()), но градиент НЕ доходит до ln_f/lm_head, а до
   x доходит.
2. chunked_boost_loss(..., own_readout_fn=frozen_readout) -> градиент НЕ
   накапливается на параметрах readout (ln_f.weight/bias, lm_head.weight), но
   h_i/alpha_i получают ненулевой градиент; own_readout_fn=None (default) —
   старое поведение, readout ТОЖЕ получает градиент (regression-проверка).
3. BLOCK_LAYER_RANGES-конструкция (k=6, block-size=2 на 12-слойной модели) даёт
   3 блока (6,8),(8,10),(10,12) — та же формула, что в train_sid_f.py/
   train_sid_p_boost.py.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.sid.chunked import chunked_boost_loss
from src.sid.forward import embed, forward_range, frozen_readout, readout


torch.manual_seed(0)
config = GPTConfig()
model = GPT(config)

B = 4
idx = torch.randint(0, config.vocab_size, (B, config.block_size))
target = torch.randint(0, config.vocab_size, (B, config.block_size))
K = 5

# --- 1. frozen_readout: значение совпадает, градиент не доходит до readout ---
h = forward_range(model, embed(model, idx), 0, K).detach()
x_a = h.clone().requires_grad_(True)
x_b = h.clone().requires_grad_(True)

out_normal = readout(model, x_a)
out_frozen = frozen_readout(model, x_b)
print(f"frozen_readout(x) == readout(x) (значения): "
      f"{torch.allclose(out_normal, out_frozen, atol=1e-5)}")
assert torch.allclose(out_normal, out_frozen, atol=1e-5)

out_normal.sum().backward()
out_frozen.sum().backward()
print(f"градиент по x (вход блока) есть в обоих случаях: "
      f"normal={x_a.grad is not None}, frozen={x_b.grad is not None}")
assert x_a.grad is not None and x_b.grad is not None
print(f"градиенты по x совпадают (frozen меняет только readout, не backprop в x): "
      f"{torch.allclose(x_a.grad, x_b.grad, atol=1e-5)}")
assert torch.allclose(x_a.grad, x_b.grad, atol=1e-5)

model.zero_grad(set_to_none=True)
out_normal2 = readout(model, h.clone().requires_grad_(True))
out_normal2.sum().backward()
ln_f_grad_normal = model.transformer.ln_f.weight.grad
lm_head_grad_normal = model.lm_head.weight.grad
print(f"обычный readout: градиент на ln_f.weight/lm_head.weight есть: "
      f"{ln_f_grad_normal is not None}, {lm_head_grad_normal is not None}")
assert ln_f_grad_normal is not None and lm_head_grad_normal is not None

model.zero_grad(set_to_none=True)
out_frozen2 = frozen_readout(model, h.clone().requires_grad_(True))
out_frozen2.sum().backward()
ln_f_grad_frozen = model.transformer.ln_f.weight.grad
lm_head_grad_frozen = model.lm_head.weight.grad
print(f"frozen_readout: градиент на ln_f.weight/lm_head.weight (ожидается None): "
      f"{ln_f_grad_frozen}, {lm_head_grad_frozen}")
assert ln_f_grad_frozen is None and lm_head_grad_frozen is None

# --- 2. chunked_boost_loss с own_readout_fn=frozen_readout ---
model.zero_grad(set_to_none=True)
h_backbone = forward_range(model, embed(model, idx), 0, K).detach()
h1 = forward_range(model, h_backbone, K, K + 1).detach()
alpha1 = torch.tensor(0.5)
h2_in = h1.detach().requires_grad_(True)
h2 = forward_range(model, h2_in, K + 1, K + 2)
alpha2 = torch.tensor(0.01, requires_grad=True)
hidden_history = [h_backbone, h1]
alpha_history = [torch.tensor(1.0), alpha1]

loss_frozen = chunked_boost_loss(model, hidden_history, alpha_history, h2, alpha2, target,
                                  chunk_size=64, own_readout_fn=frozen_readout)
loss_frozen.backward()
print(f"\nown_readout_fn=frozen_readout: grad на h2_in (вход блока): {h2_in.grad is not None}, "
      f"на alpha2: {alpha2.grad is not None and alpha2.grad.abs().item() > 0}")
assert h2_in.grad is not None
assert alpha2.grad is not None and alpha2.grad.abs().item() > 0
print(f"own_readout_fn=frozen_readout: grad на ln_f.weight (ожидается None): {model.transformer.ln_f.weight.grad}")
print(f"own_readout_fn=frozen_readout: grad на lm_head.weight (ожидается None): {model.lm_head.weight.grad}")
assert model.transformer.ln_f.weight.grad is None
assert model.lm_head.weight.grad is None

model.zero_grad(set_to_none=True)
h2_in_b = h1.detach().requires_grad_(True)
h2_b = forward_range(model, h2_in_b, K + 1, K + 2)
alpha2_b = torch.tensor(0.01, requires_grad=True)
loss_normal = chunked_boost_loss(model, hidden_history, alpha_history, h2_b, alpha2_b, target, chunk_size=64)
loss_normal.backward()
print(f"own_readout_fn=None (старое поведение): grad на ln_f.weight есть "
      f"(regression-проверка): {model.transformer.ln_f.weight.grad is not None}")
assert model.transformer.ln_f.weight.grad is not None
assert model.lm_head.weight.grad is not None

# --- 3. BLOCK_LAYER_RANGES для k=6, block-size=2, n_layer=12 ---
K3, BS, N_LAYER = 6, 2, config.n_layer
ranges = [(s, min(s + BS, N_LAYER)) for s in range(K3, N_LAYER, BS)]
print(f"\nBLOCK_LAYER_RANGES(k=6, bs=2): {ranges}")
assert ranges == [(6, 8), (8, 10), (10, 12)]

print("\nВсе проверки пройдены.")
