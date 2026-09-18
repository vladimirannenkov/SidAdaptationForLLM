"""Fit and log the complete 100% checkpoint Tuned Lens analysis."""

import json
import gc
import os
import sys
from pathlib import Path

import comet_ml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/visualization/ -> repo root
from src.data.wikitext.loader import get_batch
from src.visualization.cka import collect_cka
from src.visualization.checkpoint import load_e2e_checkpoint
from src.visualization.evaluate import evaluate_tuned_lens, make_fixed_batch_fn
from src.visualization.fit import fit_tuned_lens
from src.visualization.runtime_guard import require_memory_margin
from src.visualization.tuned_lens import TunedLens


ROOT = Path(__file__).resolve().parents[2]  # src/visualization/ -> src/ -> repo root
CHECKPOINT = ROOT / "checkpoints" / "checkpoint_100pct.pt"
ARTIFACT_DIR = ROOT / "artifacts" / "tuned_lens_100pct"
DATA_DIR = str(ROOT / "data" / "wikitext")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
FIT_STEPS = 300
EVAL_BATCHES = 20
EVAL_SEED = 1337
FIT_SEED = 20260919


def heatmap(matrix, title, xlabel, ylabel, path):
    figure, axis = plt.subplots(figsize=(12, 5))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    return figure


def layer_curve(values, title, ylabel, path):
    figure, axis = plt.subplots(figsize=(10, 5))
    depths = range(1, len(values) + 1)
    axis.plot(depths, values, marker="o")
    axis.set_title(title)
    axis.set_xlabel("Transformer depth")
    axis.set_ylabel(ylabel)
    axis.set_xticks(list(depths))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    return figure


def main():
    initial_free_ratio = require_memory_margin()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_e2e_checkpoint(str(CHECKPOINT), device=DEVICE)
    checkpoint_step = checkpoint["optimizer_step"]
    checkpoint_tokens = checkpoint["tokens_processed"]
    del checkpoint
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    lens = TunedLens(model.config.n_layer, model.config.n_embd).to(DEVICE)
    optimizer = torch.optim.AdamW(lens.parameters(), lr=1e-3, weight_decay=0.0)

    train_sampler = make_fixed_batch_fn(
        get_batch, "train", BATCH_SIZE, model.config.block_size, DEVICE, DATA_DIR, FIT_SEED
    )
    train_batch_fn = lambda split: train_sampler(split)[0]
    history = fit_tuned_lens(
        model, lens, train_batch_fn, optimizer, FIT_STEPS,
        memory_guard=require_memory_margin,
    )
    require_memory_margin()

    validation_sampler = make_fixed_batch_fn(
        get_batch, "validation", BATCH_SIZE, model.config.block_size, DEVICE, DATA_DIR, EVAL_SEED
    )
    metrics = evaluate_tuned_lens(model, lens, validation_sampler, EVAL_BATCHES)
    cka_sampler = make_fixed_batch_fn(
        get_batch, "validation", BATCH_SIZE, model.config.block_size, DEVICE, DATA_DIR, EVAL_SEED
    )
    cka = collect_cka(model, cka_sampler, EVAL_BATCHES)
    require_memory_margin()
    lens_path = ARTIFACT_DIR / "tuned_lens.pt"
    torch.save({"state_dict": lens.state_dict(), "checkpoint": str(CHECKPOINT)}, lens_path)

    experiment = comet_ml.Experiment(workspace="team-rl-exp")
    experiment.set_name("TunedLens_E2E_100pct_d272_L12_H4")
    experiment.add_tags(["TunedLens", "CKA", "E2E", "checkpoint_100pct", "wikitext103"])
    experiment.log_parameters({
        "source_checkpoint": str(CHECKPOINT), "fit_steps": FIT_STEPS,
        "fit_batch_size": BATCH_SIZE, "eval_batches": EVAL_BATCHES,
        "fit_seed": FIT_SEED, "eval_seed": EVAL_SEED, "device": DEVICE,
        "checkpoint_optimizer_step": checkpoint_step,
        "checkpoint_tokens_processed": checkpoint_tokens,
        "initial_free_ram_ratio": initial_free_ratio,
        "n_layer": model.config.n_layer, "n_embd": model.config.n_embd,
    })
    for depth, loss in enumerate(history[-1]["layer_kl"], start=1):
        experiment.log_metric(f"fit/layer_{depth}/forward_kl", loss, step=FIT_STEPS)
    for metric_name, values in metrics.means.items():
        for depth, value in enumerate(values.tolist(), start=1):
            experiment.log_metric(f"validation/layer_{depth}/{metric_name}", value)

    plots = []
    for metric_name, values in metrics.values.items():
        plots.append((
            f"{metric_name}_heatmap",
            heatmap(values.numpy(), f"Tuned Lens {metric_name}", "Position", "Layer", ARTIFACT_DIR / f"{metric_name}_heatmap.png"),
        ))
        plots.append((
            f"{metric_name}_by_layer",
            layer_curve(metrics.means[metric_name].numpy(), f"Tuned Lens {metric_name} by layer", metric_name, ARTIFACT_DIR / f"{metric_name}_by_layer.png"),
        ))
    plots.append((
        "cka_heatmap",
        heatmap(cka.numpy(), "Linear CKA between residual streams", "Layer", "Layer", ARTIFACT_DIR / "cka_heatmap.png"),
    ))
    for name, figure in plots:
        experiment.log_figure(
            figure_name=name,
            figure=figure,
            step=0,
            format="png",
            metadata={"source_checkpoint": str(CHECKPOINT), "layers": model.config.n_layer},
        )
        plt.close(figure)
    experiment.log_asset(str(lens_path), overwrite=True)
    with (ARTIFACT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({name: values.tolist() for name, values in metrics.means.items()}, handle, indent=2)
    experiment.log_asset(str(ARTIFACT_DIR / "summary.json"), overwrite=True)
    experiment.end()
    print(json.dumps({name: values.tolist() for name, values in metrics.means.items()}, indent=2))


if __name__ == "__main__":
    main()
