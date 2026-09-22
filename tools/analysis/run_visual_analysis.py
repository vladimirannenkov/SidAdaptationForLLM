"""Универсальный Tuned Lens + CKA анализ ЛЮБОГО чекпоинта проекта (E2E, SID-*,
Cascade) — не привязан к конкретному эксперименту. Строит affine Tuned Lens,
layer-wise entropy/cross-entropy/forward-KL (+ ppl-по-слоям = exp(cross_entropy)),
linear CKA между слоями, траекторию отдельных токенов, всё логируется в
отдельный Comet-эксперимент (не в тот, что писал сам training run — он уже
завершён к моменту анализа).

Архитектура модели (GPTConfig) по умолчанию берётся из checkpoint["config"]
(так хранят E2E- и SID-формат чекпоинтов, src/common/checkpoint.py и
src/sid/checkpoint.py). Cascade-чекпоинты (src/cascade/checkpoint.py) хранят
в "config" run-метаданные, а не GPTConfig — для них нужно явно передать
--model-config с путём к config.yaml эксперимента (архитектура берётся из
его секции model:)."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # известный OMP-конфликт torch+matplotlib на этой машине

import yaml

ROOT = Path(__file__).resolve().parents[2]  # tools/analysis/ -> tools/ -> repo root
sys.path.insert(0, str(ROOT))
DATA_DIR = str(ROOT / "data" / "wikitext")

import comet_ml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.wikitext.loader import get_batch
from src.model.gpt import GPT, GPTConfig
from src.visualization.cka import collect_cka
from src.visualization.evaluate import evaluate_tuned_lens, make_fixed_batch_fn
from src.visualization.fit import fit_tuned_lens
from src.visualization.token_trajectory import collect_token_trajectory
from src.visualization.tokenizer import load_tokenizer, token_label
from src.visualization.tuned_lens import TunedLens


def heatmap(matrix, title, xlabel, ylabel, path, cmap="viridis", annotate=False):
    # Keep the figure bounded for token-level matrices with hundreds of thousands
    # of evaluated positions; layer means and the dedicated 32-token trajectory
    # remain exact/full resolution.
    if matrix.shape[1] > 256:
        columns = np.linspace(0, matrix.shape[1] - 1, 256, dtype=np.int64)
        matrix = matrix[:, columns]
    figure, axis = plt.subplots(figsize=(max(10, matrix.shape[1] * 0.65), 7))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.3f}",
                          ha="center", va="center", fontsize=7, color="white")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    return figure


def token_heatmap(trajectory, tokenizer, metric_name, path):
    values = trajectory.metrics[metric_name]
    top_ids = trajectory.top_token_ids
    positions = min(32, values.shape[1])
    values = values[:, :positions]
    top_ids = top_ids[:, :positions]
    labels = [token_label(tokenizer, int(token_id)) for token_id in trajectory.input_ids[0, :positions]]
    order = list(range(values.shape[0] - 1, -1, -1))
    figure, axis = plt.subplots(figsize=(max(12, positions * 0.55), 8))
    image = axis.imshow(values[order].numpy(), aspect="auto", cmap="RdYlBu_r")
    for row, layer in enumerate(order):
        for column in range(positions):
            axis.text(column, row, token_label(tokenizer, int(top_ids[layer, column])),
                      ha="center", va="center", fontsize=7, color="black")
    axis.set_title(f"Tuned Lens {metric_name}: token trajectory")
    axis.set_xlabel("Input token")
    axis.set_ylabel("Layer (output at top)")
    axis.set_xticks(range(positions), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(order)), ["output"] + [str(layer) for layer in order[1:]])
    figure.colorbar(image, ax=axis, label=metric_name)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-config", default=None,
                         help="config.yaml эксперимента — нужен только если checkpoint['config'] НЕ является "
                              "GPTConfig (сейчас это верно только для Cascade-чекпоинтов)")
    parser.add_argument("--tags", default="", help="доп. теги через запятую (например family,k6)")
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--eval-batches", type=int, default=20)
    args = parser.parse_args()
    extra_tags = [t for t in args.tags.split(",") if t]

    checkpoint_path = Path(args.checkpoint).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if args.model_config:
        with open(args.model_config, encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)["model"]
    else:
        model_cfg = checkpoint["config"]
    model = GPT(GPTConfig(**model_cfg)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_step = checkpoint.get("optimizer_step")
    checkpoint_tokens = checkpoint.get("tokens_processed")
    del checkpoint
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    lens = TunedLens(model.config.n_layer, model.config.n_embd).to(device)
    optimizer = torch.optim.AdamW(lens.parameters(), lr=1e-3, weight_decay=0.0)
    train_sampler = make_fixed_batch_fn(get_batch, "train", 2, model.config.block_size, device, DATA_DIR, 20260919)
    history = fit_tuned_lens(model, lens, lambda split: train_sampler(split)[0], optimizer, args.fit_steps)
    validation_sampler = make_fixed_batch_fn(get_batch, "validation", 2, model.config.block_size, device, DATA_DIR, 1337)
    metrics = evaluate_tuned_lens(model, lens, validation_sampler, args.eval_batches)
    cka_sampler = make_fixed_batch_fn(get_batch, "validation", 2, model.config.block_size, device, DATA_DIR, 1337)
    cka = collect_cka(model, cka_sampler, args.eval_batches).cpu()
    trajectory_input, trajectory_target = validation_sampler("validation")
    trajectory = collect_token_trajectory(model, lens, trajectory_input[:1], trajectory_target[:1])
    tokenizer = load_tokenizer()

    lens_path = artifact_dir / "tuned_lens.pt"
    torch.save({"state_dict": lens.state_dict(), "source_checkpoint": str(checkpoint_path)}, lens_path)
    figures = []
    for metric_name, values in metrics.values.items():
        figure = heatmap(values.numpy(), f"Tuned Lens {metric_name}", "Token position", "Layer", artifact_dir / f"{metric_name}_heatmap.png", cmap="RdYlBu_r")
        figures.append((f"{metric_name}_heatmap", figure))
        layer = metrics.means[metric_name].numpy()
        curve, axis = plt.subplots(figsize=(10, 5))
        axis.plot(range(1, len(layer) + 1), layer, marker="o")
        axis.set_title(f"Tuned Lens {metric_name} by layer")
        axis.set_xlabel("Layer")
        axis.set_ylabel(metric_name)
        axis.grid(alpha=0.25)
        curve.tight_layout()
        curve.savefig(artifact_dir / f"{metric_name}_by_layer.png", dpi=160)
        figures.append((f"{metric_name}_by_layer", curve))
        figures.append((f"trajectory_{metric_name}", token_heatmap(trajectory, tokenizer, metric_name, artifact_dir / f"trajectory_{metric_name}.png")))

    # ppl-по-слоям = exp(cross_entropy по слоям) — прямое сравнение с val/ppl_block*
    # метриками, которые логируют все train-скрипты проекта.
    ppl_by_layer = np.exp(metrics.means["cross_entropy"].numpy())
    ppl_curve, ppl_axis = plt.subplots(figsize=(10, 5))
    ppl_axis.plot(range(1, len(ppl_by_layer) + 1), ppl_by_layer, marker="o", color="darkorange")
    ppl_axis.set_title("Tuned Lens PPL by layer")
    ppl_axis.set_xlabel("Layer")
    ppl_axis.set_ylabel("PPL (exp cross-entropy)")
    ppl_axis.grid(alpha=0.25)
    ppl_curve.tight_layout()
    ppl_curve.savefig(artifact_dir / "ppl_by_layer.png", dpi=160)
    figures.append(("ppl_by_layer", ppl_curve))

    figures.append(("cka_heatmap", heatmap(cka.numpy(), "Linear CKA between residual streams", "Layer", "Layer", artifact_dir / "cka_heatmap.png", cmap="viridis", annotate=True)))
    summary = {
        "source_checkpoint": str(checkpoint_path),
        "device": device,
        "checkpoint_optimizer_step": checkpoint_step,
        "checkpoint_tokens_processed": checkpoint_tokens,
        "fit_steps": args.fit_steps,
        "eval_batches": args.eval_batches,
        "metrics": {name: values.tolist() for name, values in metrics.means.items()},
        "ppl_by_layer": ppl_by_layer.tolist(),
        "fit_forward_kl": history[-1]["layer_kl"],
        "cka": cka.tolist(),
    }
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    comet_disabled = os.environ.get("COMET_MODE", "").upper() == "DISABLED"
    experiment = comet_ml.Experiment(workspace="team-rl-exp", disabled=comet_disabled)
    experiment.set_name(args.run_name)
    experiment.add_tags(["TunedLens", "CKA", "wikitext103"] + extra_tags)
    experiment.log_parameters({"source_checkpoint": str(checkpoint_path), "fit_steps": args.fit_steps, "eval_batches": args.eval_batches, "device": device, "checkpoint_optimizer_step": checkpoint_step, "checkpoint_tokens_processed": checkpoint_tokens})
    for metric_name, values in metrics.means.items():
        for layer, value in enumerate(values.tolist(), 1):
            experiment.log_metric(f"validation/layer_{layer}/{metric_name}", value)
            if metric_name == "cross_entropy":
                experiment.log_metric(f"validation/layer_{layer}/ppl", float(np.exp(value)))
    for layer, value in enumerate(history[-1]["layer_kl"], 1):
        experiment.log_metric(f"fit/layer_{layer}/forward_kl", value)
    for name, figure in figures:
        experiment.log_figure(figure_name=name, figure=figure, step=0, format="png", metadata={"source_checkpoint": str(checkpoint_path), "layers": model.config.n_layer})
        plt.close(figure)
    experiment.log_asset(str(lens_path), overwrite=True)
    experiment.log_asset(str(summary_path), overwrite=True)
    experiment.end()
    print(json.dumps({"artifact_dir": str(artifact_dir), "comet_experiment": args.run_name, "metrics": {name: values.tolist() for name, values in metrics.means.items()}}, indent=2))


if __name__ == "__main__":
    main()
