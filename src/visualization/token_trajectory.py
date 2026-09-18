"""Token-level Tuned Lens trajectories for annotated heatmaps."""

from dataclasses import dataclass

import torch

from src.visualization.metrics import cross_entropy, entropy, forward_kl
from src.visualization.model_adapter import forward_with_states


@dataclass
class TokenTrajectory:
    input_ids: torch.Tensor
    targets: torch.Tensor
    top_token_ids: torch.Tensor
    metrics: dict[str, torch.Tensor]


def collect_token_trajectory(model, lens, input_ids, targets) -> TokenTrajectory:
    model.eval()
    lens.eval()
    with torch.no_grad():
        trajectory = forward_with_states(model, input_ids)
        reference = trajectory.final_logits.float().log_softmax(-1)
        top_tokens, metric_rows = [], {"entropy": [], "cross_entropy": [], "forward_kl": []}
        for layer_index, hidden_state in enumerate(trajectory.hidden_states):
            log_probs = lens.log_probs(
                hidden_state, layer_index, model.transformer.ln_f, model.lm_head
            )
            top_tokens.append(log_probs.argmax(dim=-1))
            metric_rows["entropy"].append(entropy(log_probs))
            metric_rows["cross_entropy"].append(cross_entropy(log_probs, targets))
            metric_rows["forward_kl"].append(forward_kl(log_probs, reference))
        top_tokens.append(reference.argmax(dim=-1))
        metric_rows["entropy"].append(entropy(reference))
        metric_rows["cross_entropy"].append(cross_entropy(reference, targets))
        metric_rows["forward_kl"].append(torch.zeros_like(metric_rows["forward_kl"][0]))
    return TokenTrajectory(
        input_ids=input_ids.detach().cpu(),
        targets=targets.detach().cpu(),
        top_token_ids=torch.cat(top_tokens, dim=0).cpu(),
        metrics={name: torch.cat(rows, dim=0).cpu() for name, rows in metric_rows.items()},
    )
