"""Проверка src/common/device.py::setup_device_dtype — печатает, что реально
доступно в установленном PyTorch, и прогоняет один тестовый matmul под
выбранным autocast, чтобы выбор device/dtype не принимался молча внутри
train loop. Повторяет схему выбора device/dtype/autocast из
reference/nanogpt_pinned/train.py (pinned upstream nanoGPT)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.device import setup_device_dtype

sys.stdout.reconfigure(encoding="utf-8")

device, device_type, dtype, ptdtype, ctx = setup_device_dtype()

print(f"torch.__version__       = {torch.__version__}")
print(f"torch.version.cuda      = {torch.version.cuda}")
print(f"torch.cuda.is_available = {torch.cuda.is_available()}")
print(f"выбранный device        = {device}")
print(f"выбранный dtype         = {dtype}")

if device_type == "cuda":
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    total_mem = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
    print(f"GPU                      = {name}")
    print(f"compute capability       = sm_{major}{minor}")
    print(f"physical VRAM, GiB       = {total_mem:.3f}")

    with ctx:
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        c = a @ b
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"тестовый matmul выполнен, allocated peak = {peak:.3f} MiB, c.dtype = {c.dtype}")
else:
    print("CUDA недоступна: обучение на этой конфигурации PyTorch невозможно. "
          "Запустите этот скрипт интерпретатором CUDA-окружения (conda env py-dl).")
