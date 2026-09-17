"""SID-S с реальными верхними блоками: первый блок k=5 слоёв (с multi-token
loss), остальные 7 блоков по 1 слою с полной consistency-формулой
docs/PLAN.md §5.1: L_i = CE(y, p_i^tau) + lambda*tau^2*D_KL(q_{i-1}^tau || p_i^tau).

q_{i-1} — belief предыдущей глубины под stop-gradient; lambda линейно
разогревается от 0 до lambda_target за первые lambda_warmup_fraction токенов
фазы. CE и KL считаются chunked (src/sid/chunked.py::chunked_consistency_loss)
— полный (B,T,V) тензор не материализуется дважды одновременно.

Также логирует: val_ppl на каждой глубине (backbone + каждый верхний блок,
общий readout — честная кривая "помогает ли ещё один блок") и относительный
дрейф скрытого состояния между глубинами (независимый от KL сигнал того, не
выродился ли блок в тождественное преобразование).

Запуск (из корня репозитория):
    python src/experiments/sid_s_blocks/train.py
    python src/experiments/sid_s_blocks/train.py --smoke-test
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/experiments/sid_s_blocks/ -> repo root
from src.common.comet_logger import init_experiment
from src.common.config import apply_smoke_overrides, load_config
from src.common.device import setup_device_dtype
from src.common.lr_schedule import get_lr, set_lr
from src.data.wikitext.loader import get_raw_window
from src.model.gpt import GPT, GPTConfig
from src.sid.checkpoint import save_sid_checkpoint
from src.sid.chunked import chunked_consistency_loss
from src.sid.forward import embed, forward_range
from src.sid.losses import MAX_OFFSET, MultiTokenHeads, multi_token_loss
from src.sid.optimizers import configure_sid_optimizers

EXPERIMENT_NAME = "sid_s_blocks"
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
    m, t, data_cfg, ck, sid_cfg = cfg["model"], cfg["train"], cfg["data"], cfg["checkpoint"], cfg["sid"]

    K = sid_cfg["k"]
    TAU = sid_cfg["tau"]
    LAMBDA_TARGET = sid_cfg["lambda_target"]
    LAMBDA_WARMUP_FRACTION = sid_cfg["lambda_warmup_fraction"]

    micro_batch_size, grad_accum = t["micro_batch_size"], t["grad_accumulation_steps"]
    grad_clip, peak_lr = t["grad_clip"], t["peak_lr"]
    min_lr = peak_lr / 10
    weight_decay, betas = t["weight_decay"], tuple(t["betas"])
    min_train_metric_points = t["min_train_metric_points"]
    validation_every, validation_batches, eval_seed = t["validation_every"], t["validation_batches"], t["eval_seed"]
    target_tokens = t["target_tokens"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(ck["dir"], run_id)

    device, device_type, dtype, ptdtype, ctx = setup_device_dtype()

    config = GPTConfig(**m)
    model = GPT(config).to(device)
    heads = MultiTokenHeads(config.n_embd).to(device)

    optimizers = configure_sid_optimizers(model, K, weight_decay=weight_decay, learning_rate=peak_lr, betas=betas)
    num_upper_blocks = len(optimizers["blocks"])
    assert num_upper_blocks == config.n_layer - K
    heads_decay = [p for p in heads.parameters() if p.dim() >= 2]
    heads_nodecay = [p for p in heads.parameters() if p.dim() < 2]
    optimizers["backbone"].add_param_group({"params": heads_decay, "weight_decay": weight_decay})
    optimizers["backbone"].add_param_group({"params": heads_nodecay, "weight_decay": 0.0})

    # readout используется backbone (1 группа лоссов) + каждым из 7 верхних
    # блоков (docs/PLAN.md §4.2: усреднять накопленный градиент readout по
    # числу активных лоссов).
    num_active_readout_losses = 1 + num_upper_blocks

    total_params = sum(p.numel() for p in model.parameters())
    aux_params = sum(p.numel() for p in heads.parameters())

    tokens_per_step = micro_batch_size * grad_accum * config.block_size
    total_optimizer_steps = max(4, (target_tokens // tokens_per_step // 4) * 4)
    warmup_steps = max(1, round(0.03 * total_optimizer_steps))
    lambda_warmup_tokens = LAMBDA_WARMUP_FRACTION * target_tokens
    quarter_labels = {total_optimizer_steps * i // 4: f"{i * 25}pct" for i in (1, 2, 3, 4)}

    # Частота логирования зависит от длины прогона: минимум min_train_metric_points
    # точек и на коротком пилоте, и на полном прогоне.
    train_metrics_every = max(1, total_optimizer_steps // min_train_metric_points)

    family = "SID-S"
    experiment = init_experiment(
        cfg["comet"],
        name=f"{family}_{total_params / 1e6:.2f}M_k{K}+{num_upper_blocks}x1_consistency_{run_id}",
        tags=[family, "wikitext103", f"vocab{config.vocab_size}",
              "multi-token-backbone", "consistency-kl", f"k{K}"],
        parameters={
            "run_id": run_id, "family": family, "k": K, "num_upper_blocks": num_upper_blocks,
            "block_sizes": [K] + [1] * num_upper_blocks,
            "total_params": total_params, "aux_params": aux_params,
            "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
            "vocab_size": config.vocab_size, "block_size": config.block_size,
            "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
            "effective_batch_tokens": tokens_per_step,
            "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_optimizer_steps, "target_tokens": target_tokens,
            "tau": TAU, "lambda_target": LAMBDA_TARGET, "lambda_warmup_fraction": LAMBDA_WARMUP_FRACTION,
            "weight_decay": weight_decay, "grad_clip": grad_clip, "dtype": dtype,
            "train_metrics_every": train_metrics_every,
        },
    )
    print(f"k={K}, {num_upper_blocks} верхних блоков по 1 слою; план: {total_optimizer_steps} steps "
          f"({total_optimizer_steps * tokens_per_step:,} токенов), warmup={warmup_steps}, "
          f"train-метрики каждые {train_metrics_every} шагов, lambda warmup за {lambda_warmup_tokens:,.0f} токенов")

    def run_microbatch(tokens_processed, accumulate, collect_metrics):
        """Backbone (multi-token) -> верхние блоки (CE+KL), backward каждого сразу же."""
        raw = get_raw_window("train", micro_batch_size, config.block_size + MAX_OFFSET, device, data_cfg["data_dir"])
        y = raw[:, 1:1 + config.block_size]

        with ctx:
            h_backbone = forward_range(model, embed(model, raw[:, :config.block_size]), 0, K)
            backbone_loss, per_offset, per_weight = multi_token_loss(model, heads, h_backbone, raw, eos_id=0)
        h_prev = h_backbone.detach()
        (backbone_loss / accumulate).backward()

        lam = LAMBDA_TARGET * min(1.0, tokens_processed / max(lambda_warmup_tokens, 1))

        block_metrics = []
        for i, layer_idx in enumerate(range(K, config.n_layer)):
            h_in = h_prev.detach().requires_grad_(True)
            with ctx:
                h_i = forward_range(model, h_in, layer_idx, layer_idx + 1)
                loss_i, ce_i, kl_i = chunked_consistency_loss(model, h_prev, h_i, y, TAU, lam)
            (loss_i / accumulate).backward()
            if collect_metrics:
                with torch.no_grad():
                    drift = (h_i.detach() - h_prev).norm() / (h_prev.norm() + 1e-8)
                block_metrics.append((ce_i.item(), kl_i.item(), drift.item()))
            h_prev = h_i.detach()

        return per_offset, per_weight, block_metrics, lam

    def estimate_val_loss():
        model.eval()
        generator = torch.Generator().manual_seed(eval_seed)
        per_offset_sums = {o: 0.0 for o in (1, 2, 4)}
        block_ce_sums = [0.0] * num_upper_blocks
        block_kl_sums = [0.0] * num_upper_blocks
        block_drift_sums = [0.0] * num_upper_blocks
        with torch.no_grad():
            for _ in range(validation_batches):
                raw = get_raw_window("validation", micro_batch_size, config.block_size + MAX_OFFSET, device,
                                      data_cfg["data_dir"], generator=generator)
                y = raw[:, 1:1 + config.block_size]
                with ctx:
                    h_backbone = forward_range(model, embed(model, raw[:, :config.block_size]), 0, K)
                    _, per_offset, _ = multi_token_loss(model, heads, h_backbone, raw, eos_id=0)
                for o, v in per_offset.items():
                    per_offset_sums[o] += v
                h_prev = h_backbone
                for i, layer_idx in enumerate(range(K, config.n_layer)):
                    with ctx:
                        h_i = forward_range(model, h_prev, layer_idx, layer_idx + 1)
                        _, ce_i, kl_i = chunked_consistency_loss(model, h_prev, h_i, y, TAU, LAMBDA_TARGET)
                    drift = (h_i - h_prev).norm() / (h_prev.norm() + 1e-8)
                    block_ce_sums[i] += ce_i.item()
                    block_kl_sums[i] += kl_i.item()
                    block_drift_sums[i] += drift.item()
                    h_prev = h_i
        model.train()
        n = validation_batches
        backbone_per_offset = {o: s / n for o, s in per_offset_sums.items()}
        block_ce = [s / n for s in block_ce_sums]
        block_kl = [s / n for s in block_kl_sums]
        block_drift = [s / n for s in block_drift_sums]
        return backbone_per_offset, block_ce, block_kl, block_drift

    tokens_processed = 0
    for step in range(1, total_optimizer_steps + 1):
        t0 = time.perf_counter()
        lr = get_lr(step - 1, peak_lr, min_lr, warmup_steps, total_optimizer_steps)
        set_lr([optimizers["backbone"], optimizers["readout"]] + optimizers["blocks"], lr)

        collect = (step == 1 or step % train_metrics_every == 0)
        last_per_offset = last_per_weight = last_block_metrics = last_lam = None
        for _ in range(grad_accum):
            last_per_offset, last_per_weight, bm, last_lam = run_microbatch(tokens_processed, grad_accum, collect)
            if collect:
                last_block_metrics = bm
            tokens_processed += micro_batch_size * config.block_size

        for group in optimizers["readout"].param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad /= num_active_readout_losses

        all_trainable = list(model.parameters()) + list(heads.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, grad_clip)

        for opt in [optimizers["backbone"], optimizers["readout"]] + optimizers["blocks"]:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if device_type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - t0

        if collect:
            metrics = {
                "train/grad_norm": grad_norm.item(), "train/clipped": float(grad_norm.item() > grad_clip),
                "train/lr": lr, "train/lambda": last_lam, "train/tokens_processed": tokens_processed,
                "train/tokens_per_sec": tokens_per_step / step_time, "train/step_time_s": step_time,
            }
            for o in (1, 2, 4):
                metrics[f"train/backbone_loss_offset{o}"] = last_per_offset[o]
                metrics[f"train/backbone_weight_offset{o}"] = last_per_weight[o]
            for i, layer_idx in enumerate(range(K, config.n_layer)):
                metrics[f"train/block{layer_idx}_ce"] = last_block_metrics[i][0]
                metrics[f"train/block{layer_idx}_kl"] = last_block_metrics[i][1]
                metrics[f"train/block{layer_idx}_drift"] = last_block_metrics[i][2]
            if device_type == "cuda":
                metrics["train/vram_allocated_mib"] = torch.cuda.memory_allocated() / (1024 ** 2)
                metrics["train/vram_reserved_mib"] = torch.cuda.memory_reserved() / (1024 ** 2)
            experiment.log_metrics(metrics, step=step)
            block_str = " ".join(f"b{K+i}(ce={c:.2f},kl={k:.3f},drift={d:.3f})"
                                  for i, (c, k, d) in enumerate(last_block_metrics))
            print(f"step {step}: lambda={last_lam:.3f} {block_str} grad_norm={grad_norm.item():.3f} "
                  f"tok/s={metrics['train/tokens_per_sec']:.0f}")

        is_quarter = step in quarter_labels
        if step % validation_every == 0 or is_quarter:
            val_backbone_offset, val_block_ce, val_block_kl, val_block_drift = estimate_val_loss()
            val_metrics = {}
            for o in (1, 2, 4):
                val_metrics[f"val/backbone_loss_offset{o}"] = val_backbone_offset[o]
                val_metrics[f"val/backbone_ppl_offset{o}"] = math.exp(val_backbone_offset[o])
            for i, layer_idx in enumerate(range(K, config.n_layer)):
                val_metrics[f"val/block{layer_idx}_ce"] = val_block_ce[i]
                val_metrics[f"val/block{layer_idx}_ppl"] = math.exp(val_block_ce[i])
                val_metrics[f"val/block{layer_idx}_kl"] = val_block_kl[i]
                val_metrics[f"val/block{layer_idx}_drift"] = val_block_drift[i]
            experiment.log_metrics(val_metrics, step=step)
            depth_str = " ".join(f"b{K+i}:ppl={math.exp(c):.1f},drift={d:.3f}"
                                  for i, (c, d) in enumerate(zip(val_block_ce, val_block_drift)))
            print(f"step {step}: val backbone_ppl_offset1={math.exp(val_backbone_offset[1]):.1f} | "
                  f"глубина: {depth_str}")

            train_hparams = {
                "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
                "grad_clip": grad_clip, "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
                "weight_decay": weight_decay, "betas": betas, "tau": TAU, "lambda_target": LAMBDA_TARGET,
            }
            sid_config = {"family": family, "k": K, "block_sizes": [K] + [1] * num_upper_blocks}
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
    print(f"SID-S (k={K}, {num_upper_blocks}x1 блоков, consistency) training loop завершён")


if __name__ == "__main__":
    main()
