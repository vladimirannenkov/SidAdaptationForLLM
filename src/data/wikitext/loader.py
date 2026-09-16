"""get_batch/get_raw_window поверх бинарных token stream (см. prepare.py).

data_dir указывается явно вызывающим кодом (из config.yaml: data.data_dir),
а не выводится из расположения этого файла — раньше BIN_DIR был жёстко
привязан к директории loader.py, что ломалось при переносе файла в src/."""

import os

import numpy as np
import torch

SPLIT_FILES = {"train": "train.bin", "validation": "validation.bin", "test": "test.bin"}


def get_batch(split, batch_size, block_size, device, data_dir, generator=None):
    bin_path = os.path.join(data_dir, "bin", SPLIT_FILES[split])
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    # generator=None -> общий (глобальный) RNG обучения. Свой generator
    # (torch.Generator().manual_seed(...)) даёт одни и те же "фиксированные
    # окна" при каждом вызове (для валидации) и не расходует/не сбивает
    # общий RNG, от которого зависит resume.
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])

    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_raw_window(split, batch_size, window_size, device, data_dir, generator=None):
    """Как get_batch, но возвращает один тензор (B, window_size) без разбиения
    на x/y — нужно для multi-token backbone loss, где входу нужно ровно
    block_size токенов, а целям для дальних offset — ещё несколько сверху."""
    bin_path = os.path.join(data_dir, "bin", SPLIT_FILES[split])
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - window_size, (batch_size,), generator=generator)
    raw = torch.stack([torch.from_numpy(data[i:i + window_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        raw = raw.pin_memory().to(device, non_blocking=True)
    else:
        raw = raw.to(device)
    return raw
