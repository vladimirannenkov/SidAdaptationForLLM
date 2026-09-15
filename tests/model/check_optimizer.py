"""Проверка блока 06: реальный optimizer.step() меняет веса, группы корректны."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.gpt import GPT, GPTConfig

config = GPTConfig()
model = GPT(config)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95))

total_params = sum(p.numel() for p in model.parameters())
decay_n = sum(p.numel() for p in optimizer.param_groups[0]["params"])
nodecay_n = sum(p.numel() for p in optimizer.param_groups[1]["params"])
print(f"decay params = {decay_n:,}, nodecay params = {nodecay_n:,}, сумма = {decay_n + nodecay_n:,} "
      f"(ожидалось {total_params:,})")
print(f"weight_decay в группах: {[g['weight_decay'] for g in optimizer.param_groups]} (ожидалось [0.1, 0.0])")

idx = torch.randint(0, config.vocab_size, (2, 8))
targets = torch.randint(0, config.vocab_size, (2, 8))
used_id = idx[0, 0].item()
unused_id = (set(range(config.vocab_size)) - set(idx.flatten().tolist())).pop()

# c_fc.weight участвует в каждом forward целиком -> градиент точно ненулевой.
w_before = model.transformer.h[0].mlp.c_fc.weight[0, 0].item()
# строка wte встреченного токена -> получит и Adam-обновление, и decay.
used_before = model.transformer.wte.weight[used_id, 0].item()
# строка wte НЕвстреченного токена -> градиент 0, изменится лишь decoupled
# weight decay (на 5-6 порядков меньше, чем полный Adam-шаг).
unused_before = model.transformer.wte.weight[unused_id, 0].item()

_, loss = model(idx, targets)
loss.backward()
optimizer.step()

w_after = model.transformer.h[0].mlp.c_fc.weight[0, 0].item()
used_after = model.transformer.wte.weight[used_id, 0].item()
unused_after = model.transformer.wte.weight[unused_id, 0].item()

print(f"c_fc.weight[0,0]:       {w_before:.6f} -> {w_after:.6f}, diff={w_after - w_before:.6f}")
print(f"wte[встреченный id]:    {used_before:.6f} -> {used_after:.6f}, diff={used_after - used_before:.6f}")
print(f"wte[НЕвстреченный id]:  {unused_before:.8f} -> {unused_after:.8f}, diff={unused_after - unused_before:.8f} "
      f"(должен быть на порядки меньше — только weight decay, без градиента)")
