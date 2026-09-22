"""SID-Horizons: backbone (k слоёв) заморожен из E2E-чекпоинта — та же точка
старта, что SID-F. Верхние слои сгруппированы в блоки по --block-size слоёв;
КАЖДОМУ блоку назначен свой ОСНОВНОЙ горизонт предсказания s_i из --horizons
(по умолчанию 1/2/4/8 токенов вперёд), плюс вспомогательная next-token
коррекция с весом beta:

    L_i = CE(y_{t+1}, p_i) + beta * CE(y_{t+s_i}, q_i)

p_i = readout(h_i) (без проекции, обычный next-token из скрытого состояния
блока); q_i = readout(proj_i(h_i)) — тот же readout, но через отдельную
линейную проекцию для горизонта s_i (как offset-проекции в
src/sid/losses.py::MultiTokenHeads). Для блока с s_i=1 вспомогательный член
не нужен (совпадает с основным) — loss_i = CE(y_{t+1}, p_i).

Идея (не искусственно давить CKA между блоками, как --lambda-div в SID-F, а
архитектурно заставить блоки извлекать разные предиктивные признаки, задав
им разные основные задачи): см. обсуждение в docs/memory.md. Градиент между
блоками строго локальный — h_i = F_i(sg(h_{i-1})), как во всех SID-скриптах.

Запуск (из корня репозитория):
    python src/experiments/sid_horizons/train.py --k 4
    python src/experiments/sid_horizons/train.py --smoke-test
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/experiments/sid_horizons/ -> repo root
from src.common.checkpoint import load_checkpoint
from src.common.comet_logger import init_experiment
from src.common.config import apply_smoke_overrides, load_config
from src.common.device import setup_device_dtype
from src.common.lr_schedule import get_lr, set_lr
from src.data.wikitext.loader import get_raw_window
from src.model.gpt import GPT, GPTConfig
from src.sid.checkpoint import save_sid_checkpoint
from src.sid.cka import linear_cka
from src.sid.chunked import chunked_readout_loss
from src.sid.forward import embed, forward_range
from src.sid.optimizers import configure_sid_optimizers

EXPERIMENT_NAME = "sid_horizons"
EXPERIMENT_DIR = Path(__file__).resolve().parent


class HorizonHeads(nn.Module):
    """Линейная проекция h_i -> h_i для каждого блока, чей основной горизонт
    != 1 (offset=1 не требует проекции — readout(h_i) напрямую даёт next-token
    logits, как и в остальных SID-скриптах)."""

    def __init__(self, n_embd, horizons):
        super().__init__()
        self.proj = nn.ModuleDict({
            str(i): nn.Linear(n_embd, n_embd) for i, h in enumerate(horizons) if h != 1
        })
        for layer in self.proj.values():
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)


def masked_target(raw, offset, block_size, eos_id):
    """raw_tokens[:, offset:offset+block_size], с ignore_index=-1 там, где
    между текущей позицией и целью на offset>1 встретился EOS (не считать
    targets за концом документа — тот же принцип, что в multi_token_loss)."""
    target = raw[:, offset:offset + block_size]
    if offset == 1:
        return target
    is_eos = (raw == eos_id)
    cum = is_eos.cumsum(dim=1)
    cum_hi = cum[:, offset - 1:offset - 1 + block_size]
    cum_lo = cum[:, 0:block_size]
    valid = (cum_hi - cum_lo) == 0
    return target.masked_fill(~valid, -1)


def parse_args(sid_cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--k", type=int, default=sid_cfg["k"])
    parser.add_argument("--block-size", type=int, default=sid_cfg["block_size"])
    parser.add_argument("--source-checkpoint", default=sid_cfg["source_checkpoint"])
    parser.add_argument("--beta", type=float, default=sid_cfg["beta"])
    parser.add_argument("--horizons", type=int, nargs="+", default=sid_cfg["horizons"])
    return parser.parse_args()


def main():
    cfg = load_config(EXPERIMENT_DIR / "config.yaml")
    args = parse_args(cfg["sid"])
    cfg = load_config(args.config)
    if args.smoke_test:
        cfg = apply_smoke_overrides(cfg, EXPERIMENT_NAME)
    m, t, data_cfg, ck = cfg["model"], cfg["train"], cfg["data"], cfg["checkpoint"]
    K = args.k
    UPPER_BLOCK_SIZE = args.block_size
    BETA = args.beta
    HORIZONS = tuple(args.horizons)
    MAX_HORIZON = max(HORIZONS)

    micro_batch_size, grad_accum = t["micro_batch_size"], t["grad_accumulation_steps"]
    grad_clip, peak_lr = t["grad_clip"], t["peak_lr"]
    min_lr = peak_lr / 10
    weight_decay, betas = t["weight_decay"], tuple(t["betas"])
    min_train_metric_points = t["min_train_metric_points"]
    validation_every, validation_batches, eval_seed = t["validation_every"], t["validation_batches"], t["eval_seed"]
    target_tokens = t["target_tokens"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(ck["dir"], f"k{K}_bs{UPPER_BLOCK_SIZE}_{run_id}")

    device, device_type, dtype, ptdtype, ctx = setup_device_dtype()

    config = GPTConfig(**m)
    assert 0 < K < config.n_layer, f"k должен быть строго между 0 и n_layer={config.n_layer} (получено {K})"
    model = GPT(config)

    source = load_checkpoint(args.source_checkpoint)
    model.load_state_dict(source["model"])
    model.to(device)
    for p in model.transformer.wte.parameters():
        p.requires_grad = False
    for p in model.transformer.wpe.parameters():
        p.requires_grad = False
    for layer in model.transformer.h[:K]:
        for p in layer.parameters():
            p.requires_grad = False
    print(f"backbone заморожен из {args.source_checkpoint} "
          f"(шаг {source['optimizer_step']}, {source['tokens_processed']:,} токенов)")

    for layer in model.transformer.h[K:]:
        layer.apply(model._init_weights)
        for pn, p in layer.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    optimizers = configure_sid_optimizers(model, K, weight_decay=weight_decay, learning_rate=peak_lr, betas=betas,
                                           upper_block_size=UPPER_BLOCK_SIZE)
    num_blocks = len(optimizers["blocks"])
    assert num_blocks == len(HORIZONS), \
        f"число верхних блоков ({num_blocks}) должно совпадать с числом --horizons ({len(HORIZONS)})"
    block_layer_ranges = [(s, min(s + UPPER_BLOCK_SIZE, config.n_layer))
                           for s in range(K, config.n_layer, UPPER_BLOCK_SIZE)]

    def block_label(start, end):
        return str(start) if end - start == 1 else f"{start}to{end - 1}"

    # Отдельный optimizer для horizon_heads (не в общем "readout"): каждая
    # проекция получает градиент только от СВОЕГО блока (одно накопление на
    # microbatch), а не от всех блоков сразу, как ln_f/lm_head — делить её
    # градиент на num_active_readout_losses было бы неверно (заниженный LR).
    horizon_heads = HorizonHeads(config.n_embd, HORIZONS).to(device)
    horizon_optimizer = torch.optim.AdamW(
        [{"params": list(horizon_heads.parameters()), "weight_decay": 0.0}],
        lr=peak_lr, betas=betas, foreach=False,
    )

    # readout (ln_f+lm_head) получает вклад от каждого блока: 1 CE, если
    # s_i==1 (нет отдельного вспомогательного члена), иначе 2 CE (основной +
    # вспомогательный) — накопленный градиент усредняется по их числу.
    num_active_readout_losses = sum(1 if s == 1 else 2 for s in HORIZONS)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokens_per_step = micro_batch_size * grad_accum * config.block_size
    total_optimizer_steps = max(4, (target_tokens // tokens_per_step // 4) * 4)
    warmup_steps = max(1, round(0.03 * total_optimizer_steps))
    quarter_labels = {total_optimizer_steps * i // 4: f"{i * 25}pct" for i in (1, 2, 3, 4)}
    train_metrics_every = max(1, total_optimizer_steps // min_train_metric_points)

    family = "SID-Horizons"
    experiment = init_experiment(
        cfg["comet"],
        name=f"{family}_{total_params / 1e6:.2f}M_k{K}_bs{UPPER_BLOCK_SIZE}_h{'-'.join(map(str, HORIZONS))}_{run_id}",
        tags=[family, "wikitext103", f"vocab{config.vocab_size}", f"k{K}", f"blocksize{UPPER_BLOCK_SIZE}",
              "multi-horizon", "frozen-backbone"],
        parameters={
            "run_id": run_id, "family": family, "k": K, "upper_block_size": UPPER_BLOCK_SIZE,
            "horizons": list(HORIZONS), "beta": BETA, "source_checkpoint": args.source_checkpoint,
            "total_params": total_params, "trainable_params": trainable_params,
            "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
            "vocab_size": config.vocab_size, "block_size": config.block_size,
            "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
            "effective_batch_tokens": tokens_per_step,
            "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_optimizer_steps, "target_tokens": target_tokens,
            "weight_decay": weight_decay, "grad_clip": grad_clip, "dtype": dtype,
            "train_metrics_every": train_metrics_every,
        },
    )
    print(f"k={K}, {num_blocks} верхних блоков (block-size={UPPER_BLOCK_SIZE}), горизонты={HORIZONS}, beta={BETA}; "
          f"план: {total_optimizer_steps} steps ({total_optimizer_steps * tokens_per_step:,} токенов)")

    def block_loss(h_i, raw, horizon, proj, chunk_size=64):
        """Возвращает (loss_i, ce_next, ce_horizon | None)."""
        target_next = masked_target(raw, 1, config.block_size, eos_id=0)
        ce_next = chunked_readout_loss(model, h_i, target_next, ignore_index=-1, proj=None, chunk_size=chunk_size)
        if horizon == 1:
            return ce_next, ce_next, None
        target_horizon = masked_target(raw, horizon, config.block_size, eos_id=0)
        ce_horizon = chunked_readout_loss(model, h_i, target_horizon, ignore_index=-1, proj=proj, chunk_size=chunk_size)
        return ce_next + BETA * ce_horizon, ce_next, ce_horizon

    def run_microbatch(collect_metrics):
        raw = get_raw_window("train", micro_batch_size, config.block_size + MAX_HORIZON, device, data_cfg["data_dir"])
        with torch.no_grad(), ctx:
            h = forward_range(model, embed(model, raw[:, :config.block_size]), 0, K)

        per_block_loss, per_block_horizon_loss, per_block_cka, per_block_drift = [], [], [], []
        h_prev = h.detach()
        for i, (start, end) in enumerate(block_layer_ranges):
            h_in = h_prev.detach().requires_grad_(True)
            proj = (horizon_heads.proj[str(i)] if str(i) in horizon_heads.proj else None)
            with ctx:
                h_i = forward_range(model, h_in, start, end)
                loss_i, ce_next, ce_horizon = block_loss(h_i, raw, HORIZONS[i], proj)
            (loss_i / grad_accum).backward()
            per_block_loss.append(ce_next.item())
            per_block_horizon_loss.append(ce_horizon.item() if ce_horizon is not None else None)
            if collect_metrics:
                with torch.no_grad():
                    h_i_det = h_i.detach()
                    per_block_drift.append(((h_i_det - h_prev).norm() / (h_prev.norm() + 1e-8)).item())
                    per_block_cka.append(linear_cka(h_prev.reshape(-1, h_prev.size(-1)),
                                                     h_i_det.reshape(-1, h_i_det.size(-1))).item())
            h_prev = h_i.detach()
        return per_block_loss, per_block_horizon_loss, per_block_cka, per_block_drift

    def estimate_val_loss():
        model.eval()
        generator = torch.Generator().manual_seed(eval_seed)
        loss_sums = [0.0] * num_blocks
        horizon_sums = [0.0] * num_blocks
        cka_sums = [0.0] * num_blocks
        drift_sums = [0.0] * num_blocks
        with torch.no_grad():
            for _ in range(validation_batches):
                raw = get_raw_window("validation", micro_batch_size, config.block_size + MAX_HORIZON, device,
                                      data_cfg["data_dir"], generator=generator)
                h = forward_range(model, embed(model, raw[:, :config.block_size]), 0, K)
                h_prev = h
                for i, (start, end) in enumerate(block_layer_ranges):
                    proj = (horizon_heads.proj[str(i)] if str(i) in horizon_heads.proj else None)
                    with ctx:
                        h_i = forward_range(model, h_prev, start, end)
                        _, ce_next, ce_horizon = block_loss(h_i, raw, HORIZONS[i], proj)
                    loss_sums[i] += ce_next.item()
                    horizon_sums[i] += ce_horizon.item() if ce_horizon is not None else 0.0
                    drift_sums[i] += ((h_i - h_prev).norm() / (h_prev.norm() + 1e-8)).item()
                    cka_sums[i] += linear_cka(h_prev.reshape(-1, h_prev.size(-1)), h_i.reshape(-1, h_i.size(-1))).item()
                    h_prev = h_i
        model.train()
        n = validation_batches
        return ([s / n for s in loss_sums], [s / n for s in horizon_sums],
                [s / n for s in cka_sums], [s / n for s in drift_sums])

    tokens_processed = 0
    for step in range(1, total_optimizer_steps + 1):
        t0 = time.perf_counter()
        lr = get_lr(step - 1, peak_lr, min_lr, warmup_steps, total_optimizer_steps)
        set_lr([optimizers["readout"], horizon_optimizer] + optimizers["blocks"], lr)

        collect = step == 1 or step % train_metrics_every == 0
        step_losses = [[] for _ in range(num_blocks)]
        last_horizon_loss = last_cka = last_drift = None
        for _ in range(grad_accum):
            losses, horizon_losses, cka, drift = run_microbatch(collect)
            for i, l in enumerate(losses):
                step_losses[i].append(l)
            if collect:
                last_horizon_loss, last_cka, last_drift = horizon_losses, cka, drift
            tokens_processed += micro_batch_size * config.block_size

        for group in optimizers["readout"].param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad /= num_active_readout_losses

        all_trainable = list(model.transformer.ln_f.parameters()) + list(model.lm_head.parameters()) \
            + list(horizon_heads.parameters())
        for layer in model.transformer.h[K:]:
            all_trainable += list(layer.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, grad_clip)

        for opt in [optimizers["readout"], horizon_optimizer] + optimizers["blocks"]:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if device_type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - t0

        if collect:
            metrics = {
                "train/grad_norm": grad_norm.item(), "train/clipped": float(grad_norm.item() > grad_clip),
                "train/lr": lr, "train/tokens_processed": tokens_processed,
                "train/tokens_per_sec": tokens_per_step / step_time, "train/step_time_s": step_time,
            }
            for i, (start, end) in enumerate(block_layer_ranges):
                label = block_label(start, end)
                metrics[f"train/block{label}_ce_next"] = sum(step_losses[i]) / len(step_losses[i])
                if last_horizon_loss[i] is not None:
                    metrics[f"train/block{label}_ce_horizon{HORIZONS[i]}"] = last_horizon_loss[i]
                metrics[f"train/block{label}_cka"] = last_cka[i]
                metrics[f"train/block{label}_drift"] = last_drift[i]
            if device_type == "cuda":
                metrics["train/vram_allocated_mib"] = torch.cuda.memory_allocated() / (1024 ** 2)
                metrics["train/vram_reserved_mib"] = torch.cuda.memory_reserved() / (1024 ** 2)
            experiment.log_metrics(metrics, step=step)
            loss_str = " ".join(f"b{block_label(s, e)}={metrics[f'train/block{block_label(s, e)}_ce_next']:.3f}"
                                 for s, e in block_layer_ranges)
            print(f"step {step}: {loss_str} grad_norm={grad_norm.item():.3f} tok/s={metrics['train/tokens_per_sec']:.0f}")

        is_quarter = step in quarter_labels
        if step % validation_every == 0 or is_quarter:
            val_losses, val_horizons, val_ckas, val_drifts = estimate_val_loss()
            val_metrics = {}
            for i, (start, end) in enumerate(block_layer_ranges):
                label = block_label(start, end)
                val_metrics[f"val/block{label}_ce_next"] = val_losses[i]
                val_metrics[f"val/block{label}_ppl_next"] = math.exp(val_losses[i])
                if HORIZONS[i] != 1:
                    val_metrics[f"val/block{label}_ce_horizon{HORIZONS[i]}"] = val_horizons[i]
                val_metrics[f"val/block{label}_cka"] = val_ckas[i]
                val_metrics[f"val/block{label}_drift"] = val_drifts[i]
            experiment.log_metrics(val_metrics, step=step)
            print(f"step {step}: val " + " ".join(f"b{block_label(s, e)}_ppl={math.exp(val_losses[i]):.2f}"
                                                    for i, (s, e) in enumerate(block_layer_ranges)))

            train_hparams = {
                "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
                "grad_clip": grad_clip, "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
                "weight_decay": weight_decay, "betas": betas,
            }
            sid_config = {"family": family, "k": K, "upper_block_size": UPPER_BLOCK_SIZE,
                          "horizons": list(HORIZONS), "beta": BETA, "source_checkpoint": args.source_checkpoint}
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_sid_checkpoint(os.path.join(checkpoint_dir, "latest.pt"), model, horizon_heads, optimizers,
                                 config, sid_config, tokens_processed, step, train_hparams)
            print(f"step {step}: checkpoint сохранён ({checkpoint_dir}/latest.pt)")
            if is_quarter:
                label = quarter_labels[step]
                save_sid_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{label}.pt"), model, horizon_heads,
                                     optimizers, config, sid_config, tokens_processed, step, train_hparams)
                print(f"step {step}: контрольная точка {label} сохранена ({checkpoint_dir}/checkpoint_{label}.pt)")

    experiment.end()
    print(f"SID-Horizons (k={K}, bs={UPPER_BLOCK_SIZE}, horizons={HORIZONS}) training loop завершён")


if __name__ == "__main__":
    main()
