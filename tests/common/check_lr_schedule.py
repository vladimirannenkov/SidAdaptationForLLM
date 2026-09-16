"""Проверка блока 14: форма расписания get_lr — реальные числа для полного прогона."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.lr_schedule import get_lr

PEAK_LR = 3e-4
MIN_LR = PEAK_LR / 10
TOTAL_STEPS = 1524
WARMUP_STEPS = 46

checks = {
    "step 0 (старт warmup)": get_lr(0, PEAK_LR, MIN_LR, WARMUP_STEPS, TOTAL_STEPS),
    "step warmup-1 (конец warmup)": get_lr(WARMUP_STEPS - 1, PEAK_LR, MIN_LR, WARMUP_STEPS, TOTAL_STEPS),
    "step warmup (пик)": get_lr(WARMUP_STEPS, PEAK_LR, MIN_LR, WARMUP_STEPS, TOTAL_STEPS),
    "step на середине decay": get_lr((WARMUP_STEPS + TOTAL_STEPS) // 2, PEAK_LR, MIN_LR, WARMUP_STEPS, TOTAL_STEPS),
    "step TOTAL_STEPS (конец)": get_lr(TOTAL_STEPS, PEAK_LR, MIN_LR, WARMUP_STEPS, TOTAL_STEPS),
    "step TOTAL_STEPS+100 (после decay)": get_lr(TOTAL_STEPS + 100, PEAK_LR, MIN_LR, WARMUP_STEPS, TOTAL_STEPS),
}
for label, lr in checks.items():
    print(f"{label}: lr={lr:.6e}")

assert get_lr(0, PEAK_LR, MIN_LR, WARMUP_STEPS, TOTAL_STEPS) < PEAK_LR, "на шаге 0 LR должен быть меньше пика"
assert abs(get_lr(WARMUP_STEPS, PEAK_LR, MIN_LR, WARMUP_STEPS, TOTAL_STEPS) - PEAK_LR) < 1e-12, "на шаге warmup LR должен быть ровно пиковым"
assert abs(get_lr(TOTAL_STEPS + 1, PEAK_LR, MIN_LR, WARMUP_STEPS, TOTAL_STEPS) - MIN_LR) < 1e-12, "после decay_steps LR должен быть ровно min_lr"
mid = get_lr((WARMUP_STEPS + TOTAL_STEPS) // 2, PEAK_LR, MIN_LR, WARMUP_STEPS, TOTAL_STEPS)
assert MIN_LR < mid < PEAK_LR, "в середине decay LR должен быть строго между min и peak"
print("\nвсе проверки формы расписания пройдены")
