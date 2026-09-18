"""Layer-wise Tuned Lens evaluation and fixed WikiText batch factories."""

from dataclasses import dataclass

import torch

from src.visualization.metrics import cross_entropy, entropy, forward_kl
from src.visualization.model_adapter import forward_with_states


@dataclass
class LayerwiseMetrics:
    """Metric heatmaps with shape ``(n_layer, evaluated_positions)``."""

    values: dict[str, torch.Tensor]

    @property
    def means(self) -> dict[str, torch.Tensor]:
        return {name: torch.nanmean(value, dim=1) for name, value in self.values.items()}


def make_fixed_batch_fn(get_batch, split, batch_size, block_size, device, data_dir, seed):
    """Create a sampler whose sequence of windows is independent of global RNG."""
    generator = torch.Generator().manual_seed(seed)

    def batch_fn(_requested_split=split):
        return get_batch(split, batch_size, block_size, device, data_dir, generator=generator)

    return batch_fn


def evaluate_tuned_lens(model, lens, batch_fn, batches):
    """Collect entropy, target CE and forward KL for every layer and position."""
    model.eval()
    lens.eval()
    n_layer = len(lens.translators)
    collected = {
        name: [[] for _ in range(n_layer)]
        for name in ("entropy", "cross_entropy", "forward_kl")
    }
    with torch.no_grad():
        for _ in range(batches):
            input_ids, targets = batch_fn("validation")
            trajectory = forward_with_states(model, input_ids)
            reference = trajectory.final_logits.float().log_softmax(-1)
            for layer_index, hidden_state in enumerate(trajectory.hidden_states):
                log_probs = lens.log_probs(
                    hidden_state, layer_index, model.transformer.ln_f, model.lm_head
                )
                metric_values = {
                    "entropy": entropy(log_probs),
                    "cross_entropy": cross_entropy(log_probs, targets),
                    "forward_kl": forward_kl(log_probs, reference),
                }
                for name, value in metric_values.items():
                    collected[name][layer_index].append(value.flatten().cpu())

    values = {
        name: torch.stack([torch.cat(rows) for rows in layer_rows])
        for name, layer_rows in collected.items()
    }
    return LayerwiseMetrics(values)
