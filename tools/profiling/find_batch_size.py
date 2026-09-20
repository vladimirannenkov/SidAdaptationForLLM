"""Калибровка microbatch (PLAN.md §7.5, шаг 3): найти максимальный batch_size,
помещающийся в бюджет 6-7 ГБ, реальным запуском полного training step
(forward+backward+optimizer.step(), как в блоке 07), а не по формуле на бумаге.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig

TARGET_GIB = 6.8  # ближе к верхней границе бюджета 6-7 ГБ (PLAN.md §7.5), с запасом до 7
config = GPTConfig()


def try_batch(batch_size):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = GPT(config).cuda()
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95))
    idx = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device="cuda")
    targets = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device="cuda")
    try:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        ok = peak_gib <= TARGET_GIB
    except torch.OutOfMemoryError:
        ok, peak_gib = False, None
    finally:
        del model, optimizer
        torch.cuda.empty_cache()
    return ok, peak_gib


# 1. Экспоненциальный поиск верхней границы (удваиваем, пока помещается).
lo, b = 4, 4
while True:
    ok, peak = try_batch(b)
    print(f"batch={b:5d}  ok={ok!s:5}  peak={peak}")
    if ok:
        lo = b
        b *= 2
    else:
        hi = b
        break

# 2. Бинарный поиск точной границы между lo (помещается) и hi (не помещается).
while hi - lo > 1:
    mid = (lo + hi) // 2
    ok, peak = try_batch(mid)
    print(f"batch={mid:5d}  ok={ok!s:5}  peak={peak}")
    if ok:
        lo = mid
    else:
        hi = mid

ok, peak = try_batch(lo)
print(f"\nмаксимальный microbatch в бюджете {TARGET_GIB} GiB: {lo} (реальный пик {peak:.3f} GiB)")
print(f"токенов на microbatch: {lo * config.block_size:,}")
