"""Checkpoint Cascade-SID — свой формат (alphas + stage/stage_step вместо
optimizer_step), через общие src.common.checkpoint_io хелперы."""

import torch

from src.common.checkpoint_io import atomic_torch_save, capture_rng_state, restore_rng_state


def save_checkpoint(path, model, alphas, optimizer, stage, stage_step, tokens_processed, config):
    state = {
        "model": model.state_dict(),
        "alphas": torch.stack([alpha.detach() for alpha in alphas]).cpu(),
        "optimizer": optimizer.state_dict(),
        "stage": stage, "stage_step": stage_step,
        "tokens_processed": tokens_processed, "config": config,
        "rng_state": capture_rng_state(),
    }
    atomic_torch_save(path, state)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    restore_rng_state(state["rng_state"])
    return state
