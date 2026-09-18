"""Fit affine Tuned Lens translators against a frozen GPT checkpoint."""

from contextlib import nullcontext

import torch

from src.visualization.metrics import forward_kl
from src.visualization.model_adapter import forward_with_states


def _batch_loss(model, lens, input_ids, training):
    with torch.no_grad():
        trajectory = forward_with_states(model, input_ids)
        reference = trajectory.final_logits.float().log_softmax(-1)

    context = nullcontext() if training else torch.no_grad()
    with context:
        layer_losses = []
        for layer_index, hidden_state in enumerate(trajectory.hidden_states):
            log_probs = lens.log_probs(
                hidden_state, layer_index, model.transformer.ln_f, model.lm_head
            )
            layer_losses.append(forward_kl(log_probs, reference).mean())
        return torch.stack(layer_losses)


def fit_tuned_lens(
    model,
    lens,
    batch_fn,
    optimizer,
    train_steps: int,
    grad_clip: float = 1.0,
    memory_guard=None,
):
    """Train translators and return one mean KL value per optimizer step."""
    model.eval()
    lens.train()
    history = []
    for step in range(train_steps):
        if memory_guard is not None and step % 10 == 0:
            memory_guard()
        input_ids = batch_fn("train")
        optimizer.zero_grad(set_to_none=True)
        layer_losses = _batch_loss(model, lens, input_ids, training=True)
        loss = layer_losses.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lens.parameters(), grad_clip)
        optimizer.step()
        history.append({"loss": loss.item(), "layer_kl": layer_losses.detach().tolist()})
    return history
