"""Проверка блока 16: sid.forward даёт БИТОВО тот же результат, что GPT.forward,
и составные диапазоны (0:k)+(k:n) эквивалентны прямому (0:n)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.sid.forward import embed, forward_range, readout

torch.manual_seed(0)
config = GPTConfig()
model = GPT(config)
model.eval()  # dropout=0 в конфиге, но эталон и так детерминирован

idx = torch.randint(0, config.vocab_size, (2, 20))
targets = torch.randint(0, config.vocab_size, (2, 20))

with torch.no_grad():
    reference_logits, _ = model(idx, targets)

    # 1. embed -> forward_range(0, n_layer) -> readout должно совпасть с GPT.forward.
    x = embed(model, idx)
    x = forward_range(model, x, 0, config.n_layer)
    logits = readout(model, x)
    print(f"полный диапазон совпадает с GPT.forward: {torch.equal(reference_logits, logits)}")

    # 2. Составной диапазон (0:k) + (k:n_layer) эквивалентен (0:n_layer) напрямую.
    k = 6
    x = embed(model, idx)
    h_backbone = forward_range(model, x, 0, k)
    h_full = forward_range(model, h_backbone, k, config.n_layer)
    logits_split = readout(model, h_full)
    print(f"составной диапазон (0:{k})+({k}:{config.n_layer}) == (0:{config.n_layer}): "
          f"{torch.equal(logits, logits_split)}")

    # 3. Пустой диапазон (k=n_layer, вырожденный случай блока 15) не меняет x.
    x_before = embed(model, idx)
    x_after = forward_range(model, x_before, config.n_layer, config.n_layer)
    print(f"пустой диапазон (start==end) не меняет x: {torch.equal(x_before, x_after)}")
