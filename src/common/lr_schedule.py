"""Косинусное LR-расписание с линейным warmup (тот же приём, что в пиннутом
reference/nanogpt_pinned/train.py)."""

import math


def get_lr(step, peak_lr, min_lr, warmup_steps, decay_steps):
    if step < warmup_steps:
        return peak_lr * (step + 1) / (warmup_steps + 1)
    if step > decay_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def set_lr(optimizers, lr):
    """optimizers — один Optimizer или список/кортеж нескольких (SID-скрипты
    держат отдельные optimizer'ы на backbone/readout/каждый верхний блок).
    Раньше в каждом train-скрипте это был повторяющийся вложенный цикл
    ``for opt in [...]: for group in opt.param_groups: group["lr"] = lr``."""
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = lr
