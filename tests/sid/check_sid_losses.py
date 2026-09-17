"""Проверка блока 17: маскирование границ документов (руками посчитанный
пример) и реальный backward через multi_token_loss на настоящей модели."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.sid.forward import embed, forward_range
from src.sid.losses import MultiTokenHeads, multi_token_loss

EOS = 0

# 1. Синтетический пример с известным положением EOS — проверяем маску руками.
#    raw = [10,11,12,13,14, EOS, 20,21,22,23], block_size=6, MAX_OFFSET=4.
raw = torch.tensor([[10, 11, 12, 13, 14, EOS, 20, 21, 22, 23]])
block_size = 6
is_eos = (raw == EOS)
cum = is_eos.cumsum(dim=1)

for offset, expected_valid in [(2, [True, True, True, True, False, True]),
                                (4, [True, True, False, False, False, True])]:
    cum_hi = cum[:, offset - 1: offset - 1 + block_size]
    cum_lo = cum[:, 0:block_size]
    valid = ((cum_hi - cum_lo) == 0)[0].tolist()
    status = "OK" if valid == expected_valid else "MISMATCH"
    print(f"offset={offset}: valid={valid} expected={expected_valid} -> {status}")
    assert valid == expected_valid, f"маска для offset={offset} не совпала с ручным расчётом"

# 2. Реальный backward на настоящей модели (CPU, не мешает GPU-обучению).
config = GPTConfig()
model = GPT(config)
heads = MultiTokenHeads(config.n_embd)

B = 4
raw_tokens = torch.randint(0, config.vocab_size, (B, config.block_size + 4))
# Немного EOS для реалистичности маскирования.
raw_tokens[:, 50] = config.eos_id if hasattr(config, "eos_id") else 0

x = embed(model, raw_tokens[:, :config.block_size])
h0 = forward_range(model, x, 0, config.n_layer)  # k=n_layer: backbone = вся сеть
loss, per_offset, per_weight = multi_token_loss(model, heads, h0, raw_tokens, eos_id=0)
loss.backward()

print(f"\nper-offset losses: {per_offset}")
print(f"per-offset веса (uncertainty weighting, старт -> должны быть ~1.0 все): {per_weight}")
print(f"total loss = {loss.item():.4f}")

no_grad_backbone = [n for n, p in model.named_parameters() if p.grad is None]
no_grad_heads = [n for n, p in heads.named_parameters() if p.grad is None]
print(f"параметров backbone без градиента: {len(no_grad_backbone)} (ожидалось 0)")
print(f"параметров MultiTokenHeads без градиента: {len(no_grad_heads)} (ожидалось 0)")
print(f"log_sigma.grad ненулевой: {heads.log_sigma.grad.abs().sum().item() > 0}")
print(f"lm_head.weight.grad ненулевой: {model.lm_head.weight.grad.abs().sum().item() > 0}")
