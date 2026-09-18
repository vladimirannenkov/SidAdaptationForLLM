"""Affine Tuned Lens translators for the project's custom GPT."""

import torch
import torch.nn as nn


class TunedLens(nn.Module):
    """One affine translator per post-block residual stream."""

    def __init__(self, n_layer: int, n_embd: int):
        super().__init__()
        self.translators = nn.ModuleList(
            [nn.Linear(n_embd, n_embd) for _ in range(n_layer)]
        )
        for translator in self.translators:
            nn.init.eye_(translator.weight)
            nn.init.zeros_(translator.bias)

    def translate(self, hidden_state: torch.Tensor, layer_index: int) -> torch.Tensor:
        return self.translators[layer_index](hidden_state)

    def logits(
        self,
        hidden_state: torch.Tensor,
        layer_index: int,
        final_norm: nn.Module,
        lm_head: nn.Module,
    ) -> torch.Tensor:
        translated = self.translate(hidden_state, layer_index)
        return lm_head(final_norm(translated))

    def log_probs(
        self,
        hidden_state: torch.Tensor,
        layer_index: int,
        final_norm: nn.Module,
        lm_head: nn.Module,
    ) -> torch.Tensor:
        return self.logits(hidden_state, layer_index, final_norm, lm_head).log_softmax(-1)
