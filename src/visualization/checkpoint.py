"""Checkpoint loading for read-only layer-wise model analysis."""

import torch

from src.model.gpt import GPT, GPTConfig


def load_e2e_checkpoint(path: str, device: str = "cpu") -> tuple[GPT, dict]:
    """Load an E2E model without restoring training RNG or optimizer state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint
