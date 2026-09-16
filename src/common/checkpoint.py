"""Checkpoint E2E-baseline (обычная модель, один optimizer, без SID-полей).
SID/Cascade используют собственные форматы (src/sid/checkpoint.py,
src/cascade/checkpoint.py) через те же common.checkpoint_io хелперы, но с
другим набором полей — см. их докстринги."""

import dataclasses

import torch

from src.common.checkpoint_io import atomic_torch_save, capture_rng_state, restore_rng_state


def save_checkpoint(path, model, optimizer, config, tokens_processed, optimizer_step, train_hparams):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "tokens_processed": tokens_processed,
        "optimizer_step": optimizer_step,
        "config": dataclasses.asdict(config),
        "train_hparams": train_hparams,
    }
    atomic_torch_save(path, checkpoint)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint
