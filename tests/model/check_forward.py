"""Проверка блока 05: реальный forward + backward на GPT(GPTConfig()).

Не обучение — один шаг на случайных данных, чтобы убедиться, что граф
вычислений собирается и градиенты доходят до всех параметров, включая
раздельные wte/lm_head (проверка изоляции из блока 02 на реальном backward,
а не только через identity тензоров).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.gpt import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
config = GPTConfig()
model = GPT(config).to(device)

B, T = 4, 32
idx = torch.randint(0, config.vocab_size, (B, T), device=device)
targets = torch.randint(0, config.vocab_size, (B, T), device=device)

logits, loss = model(idx, targets)
print(f"device={device}")
print(f"logits.shape = {tuple(logits.shape)} (ожидалось ({B}, {T}, {config.vocab_size}))")
print(f"loss = {loss.item():.4f} (dtype={loss.dtype}, ожидание ~ln(vocab_size)={torch.log(torch.tensor(float(config.vocab_size))):.4f} на случайных весах)")

loss.backward()

no_grad = [n for n, p in model.named_parameters() if p.grad is None]
print(f"параметров без градиента после backward: {len(no_grad)} (ожидалось 0)")

wte_grad = model.transformer.wte.weight.grad
lm_head_grad = model.lm_head.weight.grad
print(f"wte.grad is lm_head.grad -> {wte_grad is lm_head_grad} (должно быть False)")
print(f"wte.grad и lm_head.grad совпадают поэлементно -> {torch.equal(wte_grad, lm_head_grad)} (должно быть False)")

# Инференс-ветка: targets=None должен вернуть logits только по последней позиции.
model.eval()
with torch.no_grad():
    gen_logits, gen_loss = model(idx)
print(f"inference logits.shape = {tuple(gen_logits.shape)} (ожидалось ({B}, 1, {config.vocab_size})), loss = {gen_loss}")
