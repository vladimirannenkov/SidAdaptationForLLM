"""Create corrected token trajectory and combined metric figures."""

import gc
import json
import sys
from pathlib import Path

import comet_ml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/visualization/ -> repo root
from src.data.wikitext.loader import get_batch
from src.visualization.cka import collect_cka
from src.visualization.checkpoint import load_e2e_checkpoint
from src.visualization.evaluate import evaluate_tuned_lens, make_fixed_batch_fn
from src.visualization.runtime_guard import require_memory_margin
from src.visualization.token_trajectory import collect_token_trajectory
from src.visualization.tokenizer import load_tokenizer, token_label
from src.visualization.tuned_lens import TunedLens


ROOT = Path(__file__).resolve().parents[2]  # src/visualization/ -> src/ -> repo root
CHECKPOINT = ROOT / "checkpoints" / "checkpoint_100pct.pt"
LENS_PATH = ROOT / "artifacts" / "tuned_lens_100pct" / "tuned_lens.pt"
ARTIFACT_DIR = ROOT / "artifacts" / "tuned_lens_100pct_corrected"
DATA_DIR = str(ROOT / "data" / "wikitext")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
TRAJECTORY_TOKENS = 32
EVAL_BATCHES = 20
EVAL_SEED = 1337


def token_heatmap(trajectory, tokenizer, metric_name, path):
    values = trajectory.metrics[metric_name][:, :TRAJECTORY_TOKENS]
    top_ids = trajectory.top_token_ids[:, :TRAJECTORY_TOKENS]
    labels = [token_label(tokenizer, token_id) for token_id in trajectory.input_ids[0, :TRAJECTORY_TOKENS]]
    figure, axis = plt.subplots(figsize=(max(12, len(labels) * 0.7), 8))
    image = axis.imshow(values.numpy(), aspect="auto", cmap="RdYlBu_r")
    for layer in range(values.size(0)):
        for position in range(values.size(1)):
            axis.text(position, layer, token_label(tokenizer, top_ids[layer, position]),
                      ha="center", va="center", fontsize=7, color="black")
    axis.set_title(f"Tuned Lens {metric_name}: top-1 token by layer and position")
    axis.set_xlabel("Input token")
    axis.set_ylabel("Layer")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    display_order = list(range(values.size(0) - 1, -1, -1))
    axis.clear()
    image = axis.imshow(values[display_order].numpy(), aspect="auto", cmap="RdYlBu_r")
    for row, layer in enumerate(display_order):
        for position in range(values.size(1)):
            axis.text(position, row, token_label(tokenizer, top_ids[layer, position]),
                      ha="center", va="center", fontsize=8, color="black")
    axis.set_title(f"Tuned Lens {metric_name}: output and layer-wise token trajectory")
    axis.set_xlabel("Input token")
    axis.set_ylabel("Depth")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(values.size(0)), ["output"] + [str(index) for index in range(values.size(0) - 1, 0, -1)])
    figure.colorbar(image, ax=axis, label=metric_name)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    return figure


def metric_summary_heatmap(columns, path):
    names = list(columns)
    raw = np.stack([columns[name] for name in names], axis=1)
    minimum, maximum = raw.min(axis=0), raw.max(axis=0)
    normalized = (raw - minimum) / np.maximum(maximum - minimum, 1e-12)
    figure, axis = plt.subplots(figsize=(10, 7))
    image = axis.imshow(normalized, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            axis.text(column, row, f"{raw[row, column]:.3f}", ha="center", va="center", color="white")
    axis.set_title("Layer-wise Tuned Lens metrics (column-normalized colors)")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Layer")
    axis.set_xticks(range(len(names)), names, rotation=25, ha="right")
    axis.set_yticks(range(raw.shape[0]), [str(index + 1) for index in range(raw.shape[0])])
    figure.colorbar(image, ax=axis, label="normalized within metric")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    return figure


def main():
    require_memory_margin()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_e2e_checkpoint(str(CHECKPOINT), device=DEVICE)
    del checkpoint
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    lens = TunedLens(model.config.n_layer, model.config.n_embd).to(DEVICE)
    lens.load_state_dict(torch.load(LENS_PATH, map_location=DEVICE, weights_only=True)["state_dict"])

    validation_sampler = make_fixed_batch_fn(get_batch, "validation", BATCH_SIZE, model.config.block_size, DEVICE, DATA_DIR, EVAL_SEED)
    first_input, first_target = validation_sampler("validation")
    token_trajectory = collect_token_trajectory(model, lens, first_input[:1], first_target[:1])
    validation_sampler = make_fixed_batch_fn(get_batch, "validation", BATCH_SIZE, model.config.block_size, DEVICE, DATA_DIR, EVAL_SEED)
    validation = evaluate_tuned_lens(model, lens, validation_sampler, EVAL_BATCHES)
    train_sampler = make_fixed_batch_fn(get_batch, "train", BATCH_SIZE, model.config.block_size, DEVICE, DATA_DIR, EVAL_SEED)
    train_metrics = evaluate_tuned_lens(model, lens, train_sampler, EVAL_BATCHES)
    cka_sampler = make_fixed_batch_fn(get_batch, "validation", BATCH_SIZE, model.config.block_size, DEVICE, DATA_DIR, EVAL_SEED)
    cka = collect_cka(model, cka_sampler, EVAL_BATCHES)
    require_memory_margin()

    tokenizer = load_tokenizer()
    figures = []
    for metric_name in ("entropy", "cross_entropy", "forward_kl"):
        figures.append((f"trajectory_{metric_name}", token_heatmap(token_trajectory, tokenizer, metric_name, ARTIFACT_DIR / f"trajectory_{metric_name}.png")))
    summary = {"fit_forward_kl": train_metrics.means["forward_kl"].numpy(),
               "validation_entropy": validation.means["entropy"].numpy(),
               "validation_cross_entropy": validation.means["cross_entropy"].numpy(),
               "validation_forward_kl": validation.means["forward_kl"].numpy()}
    cka_figure, cka_axis = plt.subplots(figsize=(8, 7))
    image = cka_axis.imshow(cka.numpy(), vmin=0, vmax=1, cmap="viridis")
    for row in range(cka.size(0)):
        for column in range(cka.size(1)):
            cka_axis.text(column, row, f"{cka[row, column].item():.3f}",
                          ha="center", va="center", color="white", fontsize=8)
    cka_axis.set_title("Linear CKA between residual streams")
    cka_axis.set_xlabel("Layer")
    cka_axis.set_ylabel("Layer")
    cka_axis.set_xticks(range(model.config.n_layer), range(1, model.config.n_layer + 1))
    cka_axis.set_yticks(range(model.config.n_layer), range(1, model.config.n_layer + 1))
    cka_figure.colorbar(image, ax=cka_axis, label="CKA")
    cka_figure.tight_layout()
    cka_figure.savefig(ARTIFACT_DIR / "cka_heatmap.png", dpi=160)
    figures.append(("cka_heatmap", cka_figure))

    experiment = comet_ml.Experiment(workspace="team-rl-exp")
    experiment.set_name("TunedLens_E2E_100pct_corrected_visuals_v6")
    experiment.add_tags(["TunedLens", "CKA", "token-trajectory", "corrected-visuals"])
    experiment.log_parameters({"source_checkpoint": str(CHECKPOINT), "lens_path": str(LENS_PATH), "eval_batches": EVAL_BATCHES, "batch_size": BATCH_SIZE, "trajectory_tokens": TRAJECTORY_TOKENS})
    for name, figure in figures:
        experiment.log_figure(figure_name=name, figure=figure, format="png", step=0)
        plt.close(figure)
    with (ARTIFACT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({name: values.tolist() for name, values in summary.items()}, handle, indent=2)
    experiment.log_asset(str(ARTIFACT_DIR / "summary.json"), overwrite=True)
    experiment.end()
    print(json.dumps({name: values.tolist() for name, values in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
