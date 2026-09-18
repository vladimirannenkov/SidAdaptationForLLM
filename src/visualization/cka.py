"""Linear CKA collection for residual-stream representations."""

import torch

from src.visualization.model_adapter import forward_with_states


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute linear CKA for two ``(samples, features)`` matrices."""
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    cross = x.T @ y
    x_norm = x.T @ x
    y_norm = y.T @ y
    denominator = torch.linalg.matrix_norm(x_norm) * torch.linalg.matrix_norm(y_norm)
    return (torch.linalg.matrix_norm(cross).square() / denominator.clamp_min(1e-12)).cpu()


def collect_cka(model, batch_fn, batches: int, max_positions: int = 4096) -> torch.Tensor:
    """Collect fixed validation states and return an ``(layers, layers)`` CKA matrix."""
    model.eval()
    states = [[] for _ in model.transformer.h]
    collected = 0
    with torch.no_grad():
        for _ in range(batches):
            input_ids, _ = batch_fn("validation")
            trajectory = forward_with_states(model, input_ids)
            remaining = max_positions - collected
            if remaining <= 0:
                break
            for index, hidden_state in enumerate(trajectory.hidden_states):
                states[index].append(hidden_state.reshape(-1, hidden_state.size(-1))[:remaining].cpu())
            collected += min(remaining, trajectory.hidden_states[0].numel() // trajectory.hidden_states[0].size(-1))

    matrices = [torch.cat(layer_states, dim=0)[:max_positions] for layer_states in states]
    return torch.stack([torch.stack([linear_cka(x, y) for y in matrices]) for x in matrices])
