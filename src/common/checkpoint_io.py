"""Общая часть всех checkpoint-форматов проекта (E2E, SID, Cascade — каждый
хранит разные payload-поля в src/common/checkpoint.py, src/sid/checkpoint.py,
src/cascade/checkpoint.py, см. их докстринги, почему форматы НЕ объединены
в один: разный набор optimizer'ов/модулей на формат, и это осознанно
разделено, чтобы эволюция SID-формата не могла случайно сломать resume
E2E-baseline). Но сам механизм — захват/восстановление RNG-состояния и
атомарная запись на диск — был дословно продублирован во всех форматах;
здесь он вынесен один раз.
"""

import os
import random

import numpy as np
import torch


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(path, payload):
    """torch.save во временный файл + os.replace — не оставляет битый
    checkpoint на диске, если процесс упадёт/будет убит посреди записи
    (os.replace атомарен на одной файловой системе)."""
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
