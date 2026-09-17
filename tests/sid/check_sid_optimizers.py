"""Проверка блока 15: разбиение параметров и реальная изоляция при optimizer.step().

Не просто "не упало" — строим loss, который проходит ТОЛЬКО через backbone
(embeddings + h[0:k] + readout), руками, без GPT.forward (у него пока нет
частичного прохода по слоям — это будущий блок). Шагаем ВСЕМИ optimizer'ами
и проверяем: backbone/readout изменились, все верхние блоки — нет ни на бит.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.sid.optimizers import configure_sid_optimizers, partition_parameters

config = GPTConfig()
model = GPT(config)
K = 6

# 1. Партиция покрывает все параметры ровно один раз.
groups = partition_parameters(model, K)
all_from_groups = list(groups["backbone"]) + list(groups["readout"]) + [p for block in groups["blocks"] for p in block]
all_from_model = list(model.parameters())
assert len(all_from_groups) == len(all_from_model), "число тензоров-параметров не совпадает"
assert {id(p) for p in all_from_groups} == {id(p) for p in all_from_model}, "набор параметров не совпадает"
assert len(groups["blocks"]) == config.n_layer - K, "число блоков должно быть n_layer - k"
print(f"партиция: backbone={sum(p.numel() for p in groups['backbone']):,}, "
      f"readout={sum(p.numel() for p in groups['readout']):,}, "
      f"{len(groups['blocks'])} верхних блоков по {sum(p.numel() for p in groups['blocks'][0]):,} параметров")

# 2. Снимок всех весов до шага.
before = {name: p.detach().clone() for name, p in model.named_parameters()}

optimizers = configure_sid_optimizers(model, K, weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95))

# 3. Loss "только через backbone": embeddings -> h[0:K] -> ln_f -> lm_head -> CE.
#    Верхние блоки (h[K:]) в этом графе не участвуют вообще.
idx = torch.randint(0, config.vocab_size, (2, 16))
targets = torch.randint(0, config.vocab_size, (2, 16))
pos = torch.arange(16)
x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))
for layer in model.transformer.h[:K]:
    x = layer(x)
x = model.transformer.ln_f(x)
logits = model.lm_head(x)
loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
loss.backward()

# 4. Шагаем ВСЕМИ оптимизаторами (как если бы не знали заранее, что затронуто).
optimizers["backbone"].step()
optimizers["readout"].step()
for block_opt in optimizers["blocks"]:
    block_opt.step()

# 5. Сверка: backbone/readout изменились, верхние блоки — нет.
changed = {name: not torch.equal(before[name], p.detach()) for name, p in model.named_parameters()}

backbone_names = {n for n, p in model.named_parameters() if any(p is bp for bp in groups["backbone"])}
readout_names = {n for n, p in model.named_parameters() if any(p is rp for rp in groups["readout"])}
block_names = {n for n, p in model.named_parameters()
               if any(any(p is bp for bp in block) for block in groups["blocks"])}

print(f"backbone изменился: {all(changed[n] for n in backbone_names)} (ожидалось True)")
print(f"readout изменился: {all(changed[n] for n in readout_names)} (ожидалось True)")
print(f"верхние блоки НЕ изменились: {all(not changed[n] for n in block_names)} (ожидалось True)")

# 6. Вырожденный случай k=n_layer: backbone = вся сеть, blocks=[] (SID-S,
#    PLAN.md §4.3 — один большой блок с собственным/auxiliary loss, без
#    отдельных верхних блоков вообще).
model2 = GPT(config)
full_groups = partition_parameters(model2, config.n_layer)
assert full_groups["blocks"] == [], "при k=n_layer blocks должен быть пустым"
assert sum(p.numel() for p in full_groups["backbone"]) + sum(p.numel() for p in full_groups["readout"]) == \
       sum(p.numel() for p in model2.parameters()), "backbone+readout должны покрывать всю модель при k=n_layer"
full_optimizers = configure_sid_optimizers(model2, config.n_layer, weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95))
assert full_optimizers["blocks"] == [], "при k=n_layer не должно быть ни одного block-оптимизатора"
print(f"k=n_layer (SID-S, один блок без верхних слоёв): blocks=[] -> OK, "
      f"backbone+readout покрывают все {sum(p.numel() for p in model2.parameters()):,} параметров")

# 7. upper_block_size=2 (новое): k=4, блоки по 2 слоя вместо 1 -> 4 блока
#    (слои 4-5, 6-7, 8-9, 10-11) на 12-слойной модели, не 8.
model3 = GPT(config)
K3 = 4
grouped = partition_parameters(model3, K3, upper_block_size=2)
assert len(grouped["blocks"]) == (config.n_layer - K3) // 2, "должно быть (n_layer-k)/2 блоков по 2 слоя"
for block in grouped["blocks"]:
    # каждый блок должен содержать параметры ровно 2 слоёв, не 1 и не 3.
    single_layer_params = sum(p.numel() for p in model3.transformer.h[0].parameters())
    assert sum(p.numel() for p in block) == 2 * single_layer_params, "блок должен содержать ровно 2 слоя"
all_from_grouped = (list(grouped["backbone"]) + list(grouped["readout"])
                     + [p for block in grouped["blocks"] for p in block])
assert {id(p) for p in all_from_grouped} == {id(p) for p in model3.parameters()}, \
    "partition с upper_block_size=2 должна покрывать все параметры ровно один раз"
print(f"upper_block_size=2 (k={K3}): {len(grouped['blocks'])} верхних блоков по 2 слоя каждый -> OK, "
      f"партиция покрывает все параметры ровно один раз")
