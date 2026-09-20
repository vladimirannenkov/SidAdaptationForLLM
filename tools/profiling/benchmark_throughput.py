"""Реальный замер throughput (токенов/с) на калиброванной конфигурации
(MICRO_BATCH_SIZE=64, GRAD_ACCUMULATION_STEPS=4 — блоки 08/09), чтобы
оценить время полного baseline честным измерением, а не сценарным
допущением 2-6 TFLOP/s из PLAN.md §7.4 (та оценка была для старой модели
35М/V=50257 и вообще не была измерением).
"""

import os
import sys
import time
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig

MICRO_BATCH_SIZE = 64
GRAD_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 5
TIMED_STEPS = 30
TARGET_TOKENS = 100_000_000  # PLAN.md §9.2: ориентир baseline ~100 млн токенов

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"
dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device_type == "cuda" else "float32"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

config = GPTConfig()
model = GPT(config).to(device)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95))

tokens_per_step = MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS * config.block_size


def optimizer_step():
    for _ in range(GRAD_ACCUMULATION_STEPS):
        idx = torch.randint(0, config.vocab_size, (MICRO_BATCH_SIZE, config.block_size), device=device)
        targets = torch.randint(0, config.vocab_size, (MICRO_BATCH_SIZE, config.block_size), device=device)
        with ctx:
            _, loss = model(idx, targets)
            loss = loss / GRAD_ACCUMULATION_STEPS
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


for _ in range(WARMUP_STEPS):
    optimizer_step()
if device_type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(TIMED_STEPS):
    optimizer_step()
if device_type == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

total_tokens = TIMED_STEPS * tokens_per_step
tokens_per_sec = total_tokens / elapsed
seconds_per_100m = TARGET_TOKENS / tokens_per_sec

print(f"device={device}, dtype={dtype}, micro_batch={MICRO_BATCH_SIZE}, accumulation={GRAD_ACCUMULATION_STEPS}")
print(f"{TIMED_STEPS} optimizer steps x {tokens_per_step:,} токенов/step = {total_tokens:,} токенов за {elapsed:.2f} с")
print(f"измеренный throughput: {tokens_per_sec:,.0f} токенов/с")
print(f"экстраполяция на {TARGET_TOKENS:,} токенов (PLAN.md §9.2, ориентир baseline): "
      f"{seconds_per_100m/3600:.2f} ч ({seconds_per_100m/60:.1f} мин)")
