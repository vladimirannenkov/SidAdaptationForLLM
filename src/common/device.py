"""Единая настройка device/dtype/autocast — раньше эти 6-7 строк были
буквально скопированы в каждый train-скрипт (E2E, все SID-варианты,
Newton-SID, Cascade). bfloat16 выбирается на CUDA, если поддерживается
(RTX 5060 её поддерживает — см. docs/history.md), иначе float16 на GPU
без bf16, иначе float32 на CPU. autocast — контекст-менеджер, под которым
matmul/attention считаются в этом более узком dtype, а веса/градиенты
остаются fp32 (economy памяти без ручного приведения типов).
"""

from contextlib import nullcontext

import torch


def setup_device_dtype():
    """Возвращает (device, device_type, dtype_name, torch_dtype, autocast_ctx)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    if device_type == "cuda":
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        dtype = "float32"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return device, device_type, dtype, ptdtype, ctx
