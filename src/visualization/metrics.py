"""Stable token-level metrics for Tuned Lens prediction trajectories."""

import torch


def entropy(log_probs: torch.Tensor) -> torch.Tensor:
    probs = log_probs.float().exp()
    return -(probs * log_probs.float()).sum(dim=-1)


def cross_entropy(
    log_probs: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1
) -> torch.Tensor:
    safe_targets = targets.masked_fill(targets == ignore_index, 0)
    values = -log_probs.float().gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return values.masked_fill(targets == ignore_index, float("nan"))


def forward_kl(
    log_probs: torch.Tensor, reference_log_probs: torch.Tensor
) -> torch.Tensor:
    probs = log_probs.float().exp()
    return (probs * (log_probs.float() - reference_log_probs.float())).sum(dim=-1)


def masked_mean(values: torch.Tensor) -> torch.Tensor:
    return torch.nanmean(values)

