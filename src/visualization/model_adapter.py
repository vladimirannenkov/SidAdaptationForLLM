"""Extract residual-stream states and final logits from the project GPT."""

from dataclasses import dataclass

import torch


@dataclass
class ModelTrajectory:
    """Layer states and the model's final vocabulary logits for one batch."""

    hidden_states: list[torch.Tensor]
    final_logits: torch.Tensor


def forward_with_states(model, input_ids: torch.Tensor) -> ModelTrajectory:
    """Run GPT once and return post-block states plus full final logits."""
    _, sequence_length = input_ids.shape
    if sequence_length > model.config.block_size:
        raise ValueError(
            f"sequence length {sequence_length} exceeds block_size "
            f"{model.config.block_size}"
        )

    positions = torch.arange(sequence_length, device=input_ids.device)
    x = model.transformer.drop(
        model.transformer.wte(input_ids) + model.transformer.wpe(positions)
    )
    hidden_states = []
    for block in model.transformer.h:
        x = block(x)
        hidden_states.append(x)

    final_logits = model.lm_head(model.transformer.ln_f(x))
    return ModelTrajectory(hidden_states, final_logits)

