"""Проверка блока 02: реальный подсчёт параметров GPT(GPTConfig()) — без обучения,
без forward. GPTConfig() без аргументов = зафиксированная конфигурация Stage A
(12 x 192, 4 heads, context 256, vocab 50257) из PLAN.md §7.5.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.gpt import GPT, GPTConfig

config = GPTConfig()
model = GPT(config)

# tensor.numel() — общее число элементов тензора (произведение всех размерностей).
# У nn.Parameter это число обучаемых чисел, которые хранит именно этот вес.
total = sum(p.numel() for p in model.parameters())
wte = model.transformer.wte.weight.numel()
lm_head = model.lm_head.weight.numel()
wpe = model.transformer.wpe.weight.numel()
ln_f = sum(p.numel() for p in model.transformer.ln_f.parameters())
blocks = total - wte - lm_head - wpe - ln_f

print(f"n_layer={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}, "
      f"head_dim={config.n_embd // config.n_head}, block_size={config.block_size}")
print(f"всего параметров            = {total:,}")
print(f"  wte (token embeddings)    = {wte:,}")
print(f"  lm_head (readout)         = {lm_head:,}")
print(f"  wpe (position embeddings) = {wpe:,}")
print(f"  ln_f                      = {ln_f:,}")
print(f"  12 transformer-блоков     = {blocks:,}")
print(f"доля wte+lm_head от общего  = {(wte + lm_head) / total * 100:.1f}%")
print(f"wte.weight is lm_head.weight -> {model.transformer.wte.weight is model.lm_head.weight} "
      f"(должно быть False)")
