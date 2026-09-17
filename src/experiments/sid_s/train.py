"""SID-S: k=n_layer (backbone = вся сеть, один большой блок без верхних
слоёв — вырожденный случай src/sid/optimizers.py::partition_parameters).
Backbone loss — multi-token prediction (docs/PLAN.md §5.3, offsets 1/2/4,
uncertainty weighting), посчитанный chunked'ом (src/sid/chunked.py) вместо
полного (B,T,V) тензора логитов.

Параллельно baseline_e2e, не изменяя его: отдельные optimizer'ы
(src/sid/optimizers.py), отдельный forward (src/sid/forward.py), отдельный
checkpoint-формат (src/sid/checkpoint.py). Каждый запуск пишет чекпоинты в
свою подпапку checkpoints_sid_s/{RUN_ID}/ — повторный запуск не перезаписывает
предыдущий.

Запуск (из корня репозитория):
    python src/experiments/sid_s/train.py
    python src/experiments/sid_s/train.py --smoke-test
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/experiments/sid_s/ -> repo root
from src.common.comet_logger import init_experiment
from src.common.config import apply_smoke_overrides, load_config
from src.common.device import setup_device_dtype
from src.common.lr_schedule import get_lr, set_lr
from src.data.wikitext.loader import get_raw_window
from src.model.gpt import GPT, GPTConfig
from src.sid.checkpoint import save_sid_checkpoint
from src.sid.forward import embed, forward_range
from src.sid.losses import MAX_OFFSET, OFFSETS, MultiTokenHeads, multi_token_loss
from src.sid.optimizers import configure_sid_optimizers

EXPERIMENT_NAME = "sid_s"
EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.smoke_test:
        cfg = apply_smoke_overrides(cfg, EXPERIMENT_NAME)
    m, t, data_cfg, ck = cfg["model"], cfg["train"], cfg["data"], cfg["checkpoint"]

    micro_batch_size, grad_accum = t["micro_batch_size"], t["grad_accumulation_steps"]
    grad_clip, peak_lr = t["grad_clip"], t["peak_lr"]
    min_lr = peak_lr / 10
    weight_decay, betas = t["weight_decay"], tuple(t["betas"])
    train_metrics_every, validation_every = t["train_metrics_every"], t["validation_every"]
    validation_batches, eval_seed = t["validation_batches"], t["eval_seed"]
    target_tokens = t["target_tokens"]

    # Уникальный run_id -> каждый запуск пишет чекпоинты в свою подпапку,
    # повторный запуск не перезатирает checkpoints_sid_s/latest.pt предыдущего.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(ck["dir"], run_id)

    device, device_type, dtype, ptdtype, ctx = setup_device_dtype()

    config = GPTConfig(**m)
    K = config.n_layer  # вырожденный случай: backbone = вся сеть, blocks=[]
    model = GPT(config).to(device)
    heads = MultiTokenHeads(config.n_embd).to(device)

    optimizers = configure_sid_optimizers(model, K, weight_decay=weight_decay, learning_rate=peak_lr, betas=betas)
    assert optimizers["blocks"] == [], "k=n_layer должен давать пустой список block-оптимизаторов"
    heads_decay = [p for p in heads.parameters() if p.dim() >= 2]
    heads_nodecay = [p for p in heads.parameters() if p.dim() < 2]
    optimizers["backbone"].add_param_group({"params": heads_decay, "weight_decay": weight_decay})
    optimizers["backbone"].add_param_group({"params": heads_nodecay, "weight_decay": 0.0})

    total_params = sum(p.numel() for p in model.parameters())
    aux_params = sum(p.numel() for p in heads.parameters())

    tokens_per_step = micro_batch_size * grad_accum * config.block_size
    total_optimizer_steps = max(4, (target_tokens // tokens_per_step // 4) * 4)
    warmup_steps = max(1, round(0.03 * total_optimizer_steps))
    quarter_labels = {total_optimizer_steps * i // 4: f"{i * 25}pct" for i in (1, 2, 3, 4)}

    family = "SID-S"
    experiment = init_experiment(
        cfg["comet"],
        name=f"{family}_{total_params / 1e6:.2f}M_k{K}_multitoken_uncertainty-weighted_{run_id}",
        tags=[family, "wikitext103", f"vocab{config.vocab_size}",
              "multi-token-backbone-loss", "chunked", "uncertainty-weighting"],
        parameters={
            "run_id": run_id, "family": family, "k": K, "total_params": total_params, "aux_params": aux_params,
            "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
            "vocab_size": config.vocab_size, "block_size": config.block_size,
            "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
            "effective_batch_tokens": tokens_per_step,
            "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_optimizer_steps, "target_tokens": target_tokens,
            "weight_decay": weight_decay, "grad_clip": grad_clip, "dtype": dtype,
            "offsets": list(OFFSETS), "loss_weighting": "uncertainty (Kendall et al. 2018), learned, init=1.0 each",
        },
    )
    print(f"план прогона: {total_optimizer_steps} optimizer steps "
          f"({total_optimizer_steps * tokens_per_step:,} токенов), warmup={warmup_steps}, "
          f"четверти на шагах {sorted(quarter_labels)}")

    def estimate_val_loss():
        model.eval()
        generator = torch.Generator().manual_seed(eval_seed)
        totals = []
        per_offset_sums = {o: 0.0 for o in OFFSETS}
        with torch.no_grad():
            for _ in range(validation_batches):
                raw = get_raw_window("validation", micro_batch_size, config.block_size + MAX_OFFSET, device,
                                      data_cfg["data_dir"], generator=generator)
                with ctx:
                    x = embed(model, raw[:, :config.block_size])
                    h0 = forward_range(model, x, 0, K)
                    loss, per_offset, _ = multi_token_loss(model, heads, h0, raw, eos_id=0)
                totals.append(loss.item())
                for o, v in per_offset.items():
                    per_offset_sums[o] += v
        model.train()
        n = len(totals)
        return sum(totals) / n, {o: s / n for o, s in per_offset_sums.items()}

    tokens_processed = 0
    for step in range(1, total_optimizer_steps + 1):
        t0 = time.perf_counter()
        lr = get_lr(step - 1, peak_lr, min_lr, warmup_steps, total_optimizer_steps)
        set_lr([optimizers["backbone"], optimizers["readout"]], lr)

        step_losses = []
        for _ in range(grad_accum):
            raw = get_raw_window("train", micro_batch_size, config.block_size + MAX_OFFSET, device, data_cfg["data_dir"])
            with ctx:
                x = embed(model, raw[:, :config.block_size])
                h0 = forward_range(model, x, 0, K)
                loss, per_offset, per_weight = multi_token_loss(model, heads, h0, raw, eos_id=0)
            step_losses.append(loss.item())
            (loss / grad_accum).backward()
            tokens_processed += micro_batch_size * config.block_size

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(heads.parameters()), grad_clip)
        optimizers["backbone"].step()
        optimizers["readout"].step()
        optimizers["backbone"].zero_grad(set_to_none=True)
        optimizers["readout"].zero_grad(set_to_none=True)

        if device_type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - t0

        if step == 1 or step % train_metrics_every == 0:
            metrics = {
                "train/loss": sum(step_losses) / len(step_losses),
                "train/grad_norm": grad_norm.item(),
                "train/clipped": float(grad_norm.item() > grad_clip),
                "train/lr": lr, "train/tokens_processed": tokens_processed,
                "train/tokens_per_sec": tokens_per_step / step_time, "train/step_time_s": step_time,
            }
            for o in OFFSETS:
                metrics[f"train/loss_offset{o}"] = per_offset[o]
                metrics[f"train/weight_offset{o}"] = per_weight[o]
            if device_type == "cuda":
                metrics["train/vram_allocated_mib"] = torch.cuda.memory_allocated() / (1024 ** 2)
                metrics["train/vram_reserved_mib"] = torch.cuda.memory_reserved() / (1024 ** 2)
            experiment.log_metrics(metrics, step=step)
            print(f"step {step}: loss={metrics['train/loss']:.4f} " +
                  " ".join(f"o{o}={per_offset[o]:.3f}(w={per_weight[o]:.3f})" for o in OFFSETS) +
                  f" grad_norm={grad_norm.item():.3f} tok/s={metrics['train/tokens_per_sec']:.0f}")

        is_quarter = step in quarter_labels
        if step % validation_every == 0 or is_quarter:
            val_total, val_per_offset = estimate_val_loss()
            val_metrics = {"val/loss": val_total}
            for offset in OFFSETS:
                val_metrics[f"val/loss_offset{offset}"] = val_per_offset[offset]
                val_metrics[f"val/ppl_offset{offset}"] = math.exp(val_per_offset[offset])
            experiment.log_metrics(val_metrics, step=step)
            print(f"step {step}: val_loss={val_total:.4f} " +
                  " ".join(f"val_ppl(o{o})={math.exp(val_per_offset[o]):.2f}" for o in OFFSETS))

            train_hparams = {
                "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
                "grad_clip": grad_clip, "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
                "weight_decay": weight_decay, "betas": betas,
            }
            sid_config = {"family": family, "k": K, "offset_weights": {1: 1.0, 2: 0.25, 4: 0.125}}
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_sid_checkpoint(os.path.join(checkpoint_dir, "latest.pt"), model, heads, optimizers,
                                 config, sid_config, tokens_processed, step, train_hparams)
            print(f"step {step}: checkpoint сохранён ({checkpoint_dir}/latest.pt)")
            if is_quarter:
                label = quarter_labels[step]
                save_sid_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{label}.pt"), model, heads, optimizers,
                                     config, sid_config, tokens_processed, step, train_hparams)
                print(f"step {step}: контрольная точка {label} сохранена ({checkpoint_dir}/checkpoint_{label}.pt)")

    experiment.end()
    print("SID-S (multi-token backbone loss) training loop завершён")


if __name__ == "__main__":
    main()
