"""Checkpoint/resume для SID — отдельно от src/common/checkpoint.py (E2E).
Формат SID-чекпоинтов (несколько optimizer'ов, доп. модули вроде
MultiTokenHeads) не должен иметь ни единого шанса сломать E2E resume,
поэтому это осознанно отдельный формат, а не общий код с ветвлениями внутри
— оба используют одни и те же src.common.checkpoint_io хелперы (rng
capture/restore, атомарная запись) под капотом."""

import dataclasses

import torch

from src.common.checkpoint_io import atomic_torch_save, capture_rng_state, restore_rng_state


def save_sid_checkpoint(path, model, heads, optimizers, config, sid_config,
                         tokens_processed, optimizer_step, train_hparams):
    checkpoint = {
        "model": model.state_dict(),
        "heads": heads.state_dict() if heads is not None else None,
        "optimizers": {
            "backbone": optimizers["backbone"].state_dict(),
            "readout": optimizers["readout"].state_dict(),
            "blocks": [opt.state_dict() for opt in optimizers["blocks"]],
        },
        "rng_state": capture_rng_state(),
        "tokens_processed": tokens_processed,
        "optimizer_step": optimizer_step,
        "config": dataclasses.asdict(config),
        "sid_config": sid_config,
        "train_hparams": train_hparams,
    }
    atomic_torch_save(path, checkpoint)


def load_sid_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint
