"""Проверка блока 19: chunked_readout_loss даёт тот же результат, что и
подсчёт на всей последовательности сразу, и реально экономит VRAM."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.sid.chunked import chunked_readout_loss
from src.sid.forward import embed, forward_range, readout

torch.manual_seed(0)
config = GPTConfig()
model = GPT(config)

B = 4
idx = torch.randint(0, config.vocab_size, (B, config.block_size))
target = torch.randint(0, config.vocab_size, (B, config.block_size))
target[:, ::7] = -1  # немного "замаскированных" позиций, как от границ документов

h0 = forward_range(model, embed(model, idx), 0, config.n_layer)

# 1. Совпадение с нечанкованным подсчётом (эталон).
logits_full = readout(model, h0)
loss_reference = F.cross_entropy(logits_full.reshape(-1, logits_full.size(-1)), target.reshape(-1), ignore_index=-1)
loss_chunked = chunked_readout_loss(model, h0, target, ignore_index=-1, chunk_size=64)
print(f"эталон (вся последовательность разом) = {loss_reference.item():.8f}")
print(f"chunked (по 64 позиции)                = {loss_chunked.item():.8f}")
print(f"разница = {abs(loss_reference.item() - loss_chunked.item()):.2e} (должна быть ~0)")

# 2. Совпадение градиентов по h0 (не только значения loss).
h0_a = h0.detach().clone().requires_grad_()
h0_b = h0.detach().clone().requires_grad_()
F.cross_entropy(readout(model, h0_a).reshape(-1, config.vocab_size), target.reshape(-1), ignore_index=-1).backward()
chunked_readout_loss(model, h0_b, target, ignore_index=-1, chunk_size=64).backward()
print(f"градиенты по h0 совпадают: {torch.allclose(h0_a.grad, h0_b.grad, atol=1e-5)}")

# 3. Реальная экономия VRAM на GPU (если доступна).
if torch.cuda.is_available():
    model_gpu = GPT(config).to("cuda")
    h0_gpu = forward_range(model_gpu, embed(model_gpu, idx.to("cuda")), 0, config.n_layer).detach().requires_grad_()
    target_gpu = target.to("cuda")

    torch.cuda.reset_peak_memory_stats()
    logits = readout(model_gpu, h0_gpu)
    F.cross_entropy(logits.reshape(-1, config.vocab_size), target_gpu.reshape(-1), ignore_index=-1).backward()
    peak_full = torch.cuda.max_memory_allocated() / (1024 ** 2)

    h0_gpu.grad = None
    torch.cuda.reset_peak_memory_stats()
    chunked_readout_loss(model_gpu, h0_gpu, target_gpu, ignore_index=-1, chunk_size=64).backward()
    peak_chunked = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"\nпик VRAM без чанкинга: {peak_full:.1f} MiB")
    print(f"пик VRAM с чанкингом (64): {peak_chunked:.1f} MiB")
    print(f"экономия: {(1 - peak_chunked / peak_full) * 100:.1f}%")
else:
    print("\nCUDA недоступна в этом запуске — замер VRAM пропущен")
