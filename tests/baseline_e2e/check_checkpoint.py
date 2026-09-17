"""Проверка блока 12: resume реально воспроизводит продолжение обучения,
а не просто загружается без ошибок.

Идея: обучаем модель A 3 шага, сохраняем checkpoint, продолжаем модель A
ещё 2 шага ("эталон"). Отдельно строим свежую модель B, грузим тот же
checkpoint (веса + optimizer + RNG state) и делаем на ней те же 2 шага.
Если RNG action-for-action совпадает (get_batch использует CPU RNG через
torch.randint), веса модели B после 2 шагов должны совпасть с моделью A.
"""

import os
import sys
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.data.wikitext.loader import get_batch
from src.model.gpt import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
MICRO_BATCH_SIZE, GRAD_ACCUMULATION_STEPS, GRAD_CLIP = 8, 2, 1.0
CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_check_checkpoint.pt")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "wikitext")


def run_step(model, optimizer, tokens_processed):
    for _ in range(GRAD_ACCUMULATION_STEPS):
        idx, targets = get_batch("train", MICRO_BATCH_SIZE, model.config.block_size, device, DATA_DIR)
        with ctx:
            _, loss = model(idx, targets)
        (loss / GRAD_ACCUMULATION_STEPS).backward()
        tokens_processed += MICRO_BATCH_SIZE * model.config.block_size
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return tokens_processed, loss.item()


torch.manual_seed(1234)
config = GPTConfig()
model_a = GPT(config).to(device)
optimizer_a = model_a.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95))
train_hparams = {"micro_batch_size": MICRO_BATCH_SIZE, "grad_accumulation_steps": GRAD_ACCUMULATION_STEPS}

tokens = 0
for step in range(1, 4):
    tokens, loss = run_step(model_a, optimizer_a, tokens)
save_checkpoint(CKPT_PATH, model_a, optimizer_a, config, tokens, 3, train_hparams)

# "Эталон": продолжаем ту же модель A ещё 2 шага без перерыва.
losses_a = []
for step in range(4, 6):
    tokens, loss = run_step(model_a, optimizer_a, tokens)
    losses_a.append(loss)
reference_weight = model_a.transformer.h[0].mlp.c_fc.weight.detach().clone()

# Resume: свежая модель B, грузим checkpoint, повторяем те же 2 шага.
model_b = GPT(config).to(device)
optimizer_b = model_b.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95))
ckpt = load_checkpoint(CKPT_PATH)
model_b.load_state_dict(ckpt["model"])
optimizer_b.load_state_dict(ckpt["optimizer"])
tokens_b = ckpt["tokens_processed"]

losses_b = []
for step in range(4, 6):
    tokens_b, loss = run_step(model_b, optimizer_b, tokens_b)
    losses_b.append(loss)
resumed_weight = model_b.transformer.h[0].mlp.c_fc.weight.detach().clone()

print(f"tokens_processed: эталон={tokens}, resume={tokens_b} (должны совпасть)")
print(f"loss шагов 4-5:  эталон={[f'{l:.6f}' for l in losses_a]}")
print(f"loss шагов 4-5:  resume={[f'{l:.6f}' for l in losses_b]}")
print(f"веса c_fc совпадают ровно (equal): {torch.equal(reference_weight, resumed_weight)}")
print(f"максимальная разница весов: {(reference_weight - resumed_weight).abs().max().item():.3e}")

os.remove(CKPT_PATH)
