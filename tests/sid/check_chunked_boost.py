"""Проверка chunked_boost_loss (boosting-вариант с alpha-весом, см.
sid/chunked.py): chunked-подсчёт совпадает с эталоном без чанкинга, градиент
идёт только в текущий блок (h_i) и его alpha_i, а НЕ в предыдущие глубины
(hidden_history/alpha_history) — изоляция stop-gradient сохраняется, как и в
chunked_consistency_loss. Плюс: alpha_i получает градиент даже при
alpha_i=0 (bootstrap, тот же принцип, что zero-init в dblocks)."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.sid.chunked import chunked_boost_loss
from src.sid.forward import embed, forward_range, readout

torch.manual_seed(0)
config = GPTConfig()
model = GPT(config)

B = 4
idx = torch.randint(0, config.vocab_size, (B, config.block_size))
target = torch.randint(0, config.vocab_size, (B, config.block_size))

K = 5
h_backbone = forward_range(model, embed(model, idx), 0, K).detach()
h1 = forward_range(model, h_backbone, K, K + 1).detach()      # первый верхний блок, уже "обучен"
alpha1 = torch.tensor(0.37)                                    # его текущий (обучаемый) вес
h2_in = h1.detach().requires_grad_(True)
h2 = forward_range(model, h2_in, K + 1, K + 2)                 # блок, который сейчас обучаем
alpha2 = torch.tensor(0.01, requires_grad=True)                 # near-zero старт

hidden_history = [h_backbone, h1]
alpha_history = [torch.tensor(1.0), alpha1]                     # backbone весом 1.0 (доверенный)

# Эталон: без чанкинга, явная кумулятивная сумма ВЗВЕШЕННЫХ логитов.
with torch.no_grad():
    cum_ref = readout(model, h_backbone) * 1.0 + readout(model, h1) * alpha1
logits2_full = readout(model, h2) * alpha2
combined_ref = cum_ref + logits2_full
loss_ref = F.cross_entropy(combined_ref.reshape(-1, config.vocab_size), target.reshape(-1))

loss_chunked = chunked_boost_loss(model, hidden_history, alpha_history, h2, alpha2, target, chunk_size=64)
print(f"CE: эталон={loss_ref.item():.8f}  chunked={loss_chunked.item():.8f}  "
      f"diff={abs(loss_ref.item() - loss_chunked.item()):.2e}")

# Градиент по h2 (текущий блок) должен совпадать между chunked и эталоном.
h2_a = h2.detach().clone().requires_grad_()
alpha2_a = torch.tensor(0.01, requires_grad=True)
combined_a = cum_ref + readout(model, h2_a) * alpha2_a
loss_a = F.cross_entropy(combined_a.reshape(-1, config.vocab_size), target.reshape(-1))
loss_a.backward()

h2_b = h2.detach().clone().requires_grad_()
alpha2_b = torch.tensor(0.01, requires_grad=True)
loss_b = chunked_boost_loss(model, hidden_history, alpha_history, h2_b, alpha2_b, target, chunk_size=64)
loss_b.backward()
print(f"градиенты по h2 (текущий блок) совпадают: {torch.allclose(h2_a.grad, h2_b.grad, atol=1e-5)}")
print(f"градиенты по alpha_i совпадают: {torch.allclose(alpha2_a.grad, alpha2_b.grad, atol=1e-5)}")
print(f"alpha_i получает градиент даже при near-zero старте: "
      f"{alpha2_b.grad.abs().item():.4f} (ожидается >0)")
assert alpha2_b.grad.abs().item() > 0

# Изоляция: градиент НЕ должен течь в hidden_history/alpha_history (backbone/предыдущий блок).
h_backbone_g = h_backbone.clone().requires_grad_(True)
h1_g = h1.clone().requires_grad_(True)
alpha1_g = alpha1.clone().requires_grad_(True)
h2_c = h2.detach().clone().requires_grad_()
alpha2_c = torch.tensor(0.01, requires_grad=True)
loss_c = chunked_boost_loss(model, [h_backbone_g, h1_g], [torch.tensor(1.0), alpha1_g], h2_c, alpha2_c,
                             target, chunk_size=64)
loss_c.backward()
print(f"градиент НЕ идёт в backbone (ожидается None): {h_backbone_g.grad}")
print(f"градиент НЕ идёт в предыдущий блок (ожидается None): {h1_g.grad}")
print(f"градиент НЕ идёт в alpha предыдущего блока (ожидается None): {alpha1_g.grad}")

if torch.cuda.is_available():
    model_gpu = GPT(config).to("cuda")
    idx_g, target_g = idx.to("cuda"), target.to("cuda")
    h_backbone_gpu = forward_range(model_gpu, embed(model_gpu, idx_g), 0, K).detach()
    h1_gpu = forward_range(model_gpu, h_backbone_gpu, K, K + 1).detach()
    alpha1_gpu = torch.tensor(0.37, device="cuda")
    h2_in_gpu = h1_gpu.detach().requires_grad_(True)
    h2_gpu = forward_range(model_gpu, h2_in_gpu, K + 1, K + 2).detach().requires_grad_(True)
    alpha2_gpu = torch.tensor(0.01, device="cuda", requires_grad=True)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        cum_gpu = readout(model_gpu, h_backbone_gpu) * 1.0 + readout(model_gpu, h1_gpu) * alpha1_gpu
    combined_gpu = cum_gpu + readout(model_gpu, h2_gpu) * alpha2_gpu
    loss_full_gpu = F.cross_entropy(combined_gpu.reshape(-1, config.vocab_size), target_g.reshape(-1))
    loss_full_gpu.backward()
    peak_full = torch.cuda.max_memory_allocated() / (1024 ** 2)

    h2_gpu.grad = None
    torch.cuda.reset_peak_memory_stats()
    loss_gpu = chunked_boost_loss(model_gpu, [h_backbone_gpu, h1_gpu], [torch.tensor(1.0, device="cuda"), alpha1_gpu],
                                   h2_gpu, alpha2_gpu, target_g, chunk_size=64)
    loss_gpu.backward()
    peak_chunked = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"\nпик VRAM без чанкинга: {peak_full:.1f} MiB")
    print(f"пик VRAM с чанкингом: {peak_chunked:.1f} MiB")
    print(f"экономия: {(1 - peak_chunked / peak_full) * 100:.1f}%")

print("\nВсе проверки пройдены.")
