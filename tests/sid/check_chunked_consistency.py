"""Проверка правки блока 22: chunked_consistency_loss == нечанкованный расчёт,
и реальная экономия VRAM (два тензора (B,T,V) вместо одного не должны
одновременно жить целиком)."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.sid.chunked import chunked_consistency_loss
from src.sid.forward import embed, forward_range, readout

torch.manual_seed(0)
config = GPTConfig()
model = GPT(config)

B = 4
idx = torch.randint(0, config.vocab_size, (B, config.block_size))
target = torch.randint(0, config.vocab_size, (B, config.block_size))
TAU, LAM = 1.0, 0.5

h_prev = forward_range(model, embed(model, idx), 0, 5).detach()
h_cur = forward_range(model, h_prev, 5, 6)

# Эталон: без чанкинга.
logits_cur_full = readout(model, h_cur)
ce_ref = F.cross_entropy(logits_cur_full.reshape(-1, config.vocab_size), target.reshape(-1))
with torch.no_grad():
    prev_probs_full = F.softmax(readout(model, h_prev) / TAU, dim=-1)
log_p_cur_full = F.log_softmax(logits_cur_full / TAU, dim=-1)
kl_ref = F.kl_div(log_p_cur_full.reshape(-1, config.vocab_size), prev_probs_full.reshape(-1, config.vocab_size),
                   reduction="batchmean")
loss_ref = ce_ref + LAM * (TAU ** 2) * kl_ref

loss_chunked, ce_chunked, kl_chunked = chunked_consistency_loss(model, h_prev, h_cur, target, TAU, LAM, chunk_size=64)

print(f"CE:  эталон={ce_ref.item():.8f}  chunked={ce_chunked.item():.8f}  diff={abs(ce_ref.item()-ce_chunked.item()):.2e}")
print(f"KL:  эталон={kl_ref.item():.8f}  chunked={kl_chunked.item():.8f}  diff={abs(kl_ref.item()-kl_chunked.item()):.2e}")
print(f"loss: эталон={loss_ref.item():.8f}  chunked={loss_chunked.item():.8f}")

# Градиенты по h_cur должны совпадать.
h_cur_a = h_cur.detach().clone().requires_grad_()
h_cur_b = h_cur.detach().clone().requires_grad_()
logits_a = readout(model, h_cur_a)
ce_a = F.cross_entropy(logits_a.reshape(-1, config.vocab_size), target.reshape(-1))
kl_a = F.kl_div(F.log_softmax(logits_a / TAU, dim=-1).reshape(-1, config.vocab_size),
                prev_probs_full.reshape(-1, config.vocab_size), reduction="batchmean")
(ce_a + LAM * (TAU ** 2) * kl_a).backward()
loss_b, _, _ = chunked_consistency_loss(model, h_prev, h_cur_b, target, TAU, LAM, chunk_size=64)
loss_b.backward()
print(f"градиенты по h_cur совпадают: {torch.allclose(h_cur_a.grad, h_cur_b.grad, atol=1e-5)}")

if torch.cuda.is_available():
    model_gpu = GPT(config).to("cuda")
    idx_g, target_g = idx.to("cuda"), target.to("cuda")
    h_prev_g = forward_range(model_gpu, embed(model_gpu, idx_g), 0, 5).detach()
    h_cur_g = forward_range(model_gpu, h_prev_g, 5, 6).detach().requires_grad_()

    torch.cuda.reset_peak_memory_stats()
    logits_g = readout(model_gpu, h_cur_g)
    with torch.no_grad():
        prev_probs_g = F.softmax(readout(model_gpu, h_prev_g) / TAU, dim=-1)
    ce_g = F.cross_entropy(logits_g.reshape(-1, config.vocab_size), target_g.reshape(-1))
    kl_g = F.kl_div(F.log_softmax(logits_g / TAU, dim=-1).reshape(-1, config.vocab_size),
                     prev_probs_g.reshape(-1, config.vocab_size), reduction="batchmean")
    (ce_g + LAM * (TAU ** 2) * kl_g).backward()
    peak_full = torch.cuda.max_memory_allocated() / (1024 ** 2)

    h_cur_g.grad = None
    torch.cuda.reset_peak_memory_stats()
    loss_g, _, _ = chunked_consistency_loss(model_gpu, h_prev_g, h_cur_g, target_g, TAU, LAM, chunk_size=64)
    loss_g.backward()
    peak_chunked = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"\nпик VRAM без чанкинга (CE+KL): {peak_full:.1f} MiB")
    print(f"пик VRAM с чанкингом (CE+KL): {peak_chunked:.1f} MiB")
    print(f"экономия: {(1 - peak_chunked / peak_full) * 100:.1f}%")
