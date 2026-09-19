"""Newton-SID: backbone (k слоёв) заморожен (из уже обученного E2E-чекпоинта),
верхние блоки обучаются boosting-конструкцией (та же кумулятивная сумма
alpha_i*readout(h_i), что в sid_p_boost/train.py), но с дополнительным
вспомогательным Newton-лоссом на "сырую" (до умножения на alpha_i) коррекцию
блока dz_i (src/sid/chunked.py::chunked_boost_newton_loss, src/sid/newton.py).

Идея: dz_i = readout(h_i) - sg(z_prev) обучается через точную квадратичную
Тейлор-аппроксимацию CE вокруг z_prev (g_prev=p_prev-y, H_prev=diag(p_prev)-
p_prev*p_prev^T, оба detached) — минимум этой квадратичной формы есть ровно
Newton-шаг -H^-1*g, но обратная величина никогда не вычисляется явно.
alpha_i (src/sid/losses.py::BoostWeights) по-прежнему учится только через
настоящий CE, Newton-член учит направление коррекции независимо от alpha_i.

L_i = CE(target, softmax(sg(z_prev) + alpha_i*dz_i)) + beta*L_i^quad(dz_i)

Запуск (из корня репозитория):
    python src/experiments/newton_sid/train.py --k 6 --freeze-backbone
    python src/experiments/newton_sid/train.py --smoke-test
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/experiments/newton_sid/ -> repo root
from src.common.checkpoint import load_checkpoint
from src.common.comet_logger import init_experiment
from src.common.config import apply_smoke_overrides, load_config
from src.common.device import setup_device_dtype
from src.common.lr_schedule import get_lr, set_lr
from src.data.wikitext.loader import get_raw_window
from src.model.gpt import GPT, GPTConfig
from src.sid.cka import linear_cka
from src.sid.checkpoint import load_sid_checkpoint, save_sid_checkpoint
from src.sid.chunked import chunked_boost_newton_loss
from src.sid.forward import embed, forward_range, frozen_readout
from src.sid.losses import MAX_OFFSET, BoostWeights, MultiTokenHeads, multi_token_loss
from src.sid.optimizers import configure_sid_optimizers

EXPERIMENT_NAME = "newton_sid"
EXPERIMENT_DIR = Path(__file__).resolve().parent


class HeadsAndWeights(nn.Module):
    """Обёртка только для чекпоинтинга (как в sid_p_boost/train.py)."""

    def __init__(self, heads, boost_weights):
        super().__init__()
        self.heads = heads
        self.boost_weights = boost_weights


def parse_args(sid_cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--k", type=int, default=sid_cfg["k"])
    parser.add_argument("--source-checkpoint", default=sid_cfg["source_checkpoint"])
    parser.add_argument("--block-size", type=int, default=sid_cfg["block_size"])
    parser.add_argument("--freeze-readout", choices=["none", "except-last"], default=sid_cfg["freeze_readout"])
    parser.add_argument("--freeze-backbone-from", default=sid_cfg["freeze_backbone_from"],
                         help="SID-чекпоинт с уже обученными backbone+readout, оба заморожены целиком; "
                              "взаимоисключимо с --freeze-backbone")
    parser.add_argument("--freeze-backbone", action="store_true", default=sid_cfg["freeze_backbone"],
                         help="backbone из --source-checkpoint заморожен, readout остаётся обучаемым (основной режим)")
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--newton-lambda", type=float, default=sid_cfg["newton_lambda"])
    parser.add_argument("--newton-beta", type=float, default=sid_cfg["newton_beta"])
    parser.add_argument("--include-curvature", type=lambda s: s.lower() != "false", default=sid_cfg["include_curvature"])
    parser.add_argument("--trust-region", type=float, default=sid_cfg["trust_region"])
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--grad-accumulation-steps", type=int, default=None)
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
    FREEZE_READOUT_MODE = args.freeze_readout
    assert not (args.freeze_backbone_from is not None and args.freeze_backbone), \
        "--freeze-backbone-from и --freeze-backbone взаимоисключимы"
    FREEZE_BACKBONE_FULL = args.freeze_backbone_from is not None
    FREEZE_BACKBONE_ONLY = args.freeze_backbone
    FREEZE_BACKBONE = FREEZE_BACKBONE_FULL or FREEZE_BACKBONE_ONLY
    NEWTON_LAMBDA, NEWTON_BETA = args.newton_lambda, args.newton_beta
    INCLUDE_CURVATURE, TRUST_REGION = args.include_curvature, args.trust_region

    micro_batch_size = args.micro_batch_size if args.micro_batch_size is not None else t["micro_batch_size"]
    grad_accum = args.grad_accumulation_steps if args.grad_accumulation_steps is not None else t["grad_accumulation_steps"]
    grad_clip, peak_lr = t["grad_clip"], t["peak_lr"]
    min_lr = peak_lr / 10
    weight_decay, betas = t["weight_decay"], tuple(t["betas"])
    min_train_metric_points = t["min_train_metric_points"]
    validation_every, validation_batches, eval_seed = t["validation_every"], t["validation_batches"], t["eval_seed"]
    target_tokens = args.target_tokens if args.target_tokens is not None else t["target_tokens"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_parts = []
    if not INCLUDE_CURVATURE:
        tag_parts.append("residual")
    if TRUST_REGION is not None:
        tag_parts.append(f"trust{TRUST_REGION}")
    if FREEZE_READOUT_MODE != "none":
        tag_parts.append("frzreadout")
    if FREEZE_BACKBONE:
        tag_parts.append("frzbackbone")
    TAG = "_".join(tag_parts) if tag_parts else "base"
    checkpoint_dir = os.path.join(ck["dir"], f"k{K}_bs{UPPER_BLOCK_SIZE}_{TAG}_{run_id}")

    device, device_type, dtype, ptdtype, ctx = setup_device_dtype()

    config = GPTConfig(**m)
    assert 0 < K < config.n_layer, f"k должен быть строго между 0 и n_layer={config.n_layer} (получено {K})"
    model = GPT(config)

    if FREEZE_BACKBONE_FULL:
        sid_ckpt = load_sid_checkpoint(args.freeze_backbone_from)
        model.load_state_dict(sid_ckpt["model"])
        model.to(device)
        for p in model.transformer.wte.parameters():
            p.requires_grad = False
        for p in model.transformer.wpe.parameters():
            p.requires_grad = False
        for layer in model.transformer.h[:K]:
            for p in layer.parameters():
                p.requires_grad = False
        for p in model.transformer.ln_f.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False
        heads = None
        print(f"backbone+readout загружены из {args.freeze_backbone_from} и ПОЛНОСТЬЮ ЗАМОРОЖЕНЫ")
    elif FREEZE_BACKBONE_ONLY:
        # Backbone (embeddings + h[:K]) — из обычного E2E-чекпоинта, заморожен.
        # Readout грузится ИЗ ТОГО ЖЕ чекпоинта, но НЕ замораживается — учится
        # вместе с блоками (основной режим Newton-SID).
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
        heads = None
        print(f"backbone заморожен из {args.source_checkpoint}; readout остаётся ОБУЧАЕМЫМ")
    else:
        source = load_checkpoint(args.source_checkpoint)
        model.load_state_dict(source["model"])
        model.to(device)
        heads = MultiTokenHeads(config.n_embd).to(device)
        print(f"backbone инициализирован из {args.source_checkpoint}, НЕ заморожен")

    for layer in model.transformer.h[K:]:
        layer.apply(model._init_weights)
        for pn, p in layer.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    optimizers = configure_sid_optimizers(model, K, weight_decay=weight_decay, learning_rate=peak_lr, betas=betas,
                                           upper_block_size=UPPER_BLOCK_SIZE)
    num_upper_blocks = len(optimizers["blocks"])
    assert num_upper_blocks == -(-(config.n_layer - K) // UPPER_BLOCK_SIZE)

    block_layer_ranges = [(s, min(s + UPPER_BLOCK_SIZE, config.n_layer))
                           for s in range(K, config.n_layer, UPPER_BLOCK_SIZE)]
    assert len(block_layer_ranges) == num_upper_blocks

    def block_label(start, end):
        return str(start) if end - start == 1 else f"{start}to{end - 1}"

    boost_weights = BoostWeights(num_upper_blocks).to(device)

    if heads is not None:
        heads_decay = [p for p in heads.parameters() if p.dim() >= 2]
        heads_nodecay = [p for p in heads.parameters() if p.dim() < 2]
        optimizers["backbone"].add_param_group({"params": heads_decay, "weight_decay": weight_decay})
        optimizers["backbone"].add_param_group({"params": heads_nodecay, "weight_decay": 0.0})
    optimizers["backbone"].add_param_group({"params": list(boost_weights.parameters()), "weight_decay": 0.0})

    _backbone_readout_terms = 0 if FREEZE_BACKBONE else 1
    _upper_readout_terms = 1 if FREEZE_READOUT_MODE == "except-last" else num_upper_blocks
    num_active_readout_losses = max(1, _backbone_readout_terms + _upper_readout_terms)

    total_params = sum(p.numel() for p in model.parameters())
    aux_params = sum(p.numel() for p in heads.parameters()) if heads is not None else 0

    tokens_per_step = micro_batch_size * grad_accum * config.block_size
    total_optimizer_steps = max(4, (target_tokens // tokens_per_step // 4) * 4)
    warmup_steps = max(1, round(0.03 * total_optimizer_steps))
    quarter_labels = {total_optimizer_steps * i // 4: f"{i * 25}pct" for i in (1, 2, 3, 4)}
    train_metrics_every = max(1, total_optimizer_steps // min_train_metric_points)

    family = "SID-Newton"
    tags = [family, "wikitext103", f"vocab{config.vocab_size}", "newton", f"k{K}", f"blocksize{UPPER_BLOCK_SIZE}"]
    tags += ["multi-token-backbone", "pretrained-backbone-unfrozen"] if not FREEZE_BACKBONE else ["frozen-backbone-readout"]
    if FREEZE_READOUT_MODE != "none":
        tags.append(f"freeze-readout-{FREEZE_READOUT_MODE}")
    if not INCLUDE_CURVATURE:
        tags.append("residual-only")
    if TRUST_REGION is not None:
        tags.append("trust-region")
    experiment = init_experiment(
        cfg["comet"],
        name=f"{family}_{total_params / 1e6:.2f}M_k{K}_bs{UPPER_BLOCK_SIZE}_{TAG}_{run_id}",
        tags=tags,
        parameters={
            "run_id": run_id, "family": family, "k": K, "num_upper_blocks": num_upper_blocks,
            "upper_block_size": UPPER_BLOCK_SIZE, "block_layer_ranges": block_layer_ranges,
            "source_checkpoint": None if FREEZE_BACKBONE else args.source_checkpoint,
            "freeze_backbone_from": args.freeze_backbone_from, "backbone_frozen": FREEZE_BACKBONE,
            "readout_frozen_mode": FREEZE_READOUT_MODE,
            "newton_lambda": NEWTON_LAMBDA, "newton_beta": NEWTON_BETA,
            "include_curvature": INCLUDE_CURVATURE, "trust_region": TRUST_REGION,
            "upper_block_loss": "boosting + newton-quadratic",
            "total_params": total_params, "aux_params": aux_params,
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
    print(f"k={K}, {num_upper_blocks} верхних блоков (Newton-SID: curvature={INCLUDE_CURVATURE}, "
          f"lambda={NEWTON_LAMBDA}, beta={NEWTON_BETA}); план: {total_optimizer_steps} steps "
          f"({total_optimizer_steps * tokens_per_step:,} токенов)")

    def run_microbatch(accumulate, collect_metrics):
        raw = get_raw_window("train", micro_batch_size, config.block_size + MAX_OFFSET, device, data_cfg["data_dir"])
        y = raw[:, 1:1 + config.block_size]

        if FREEZE_BACKBONE:
            with torch.no_grad(), ctx:
                h_backbone = forward_range(model, embed(model, raw[:, :config.block_size]), 0, K)
            per_offset, per_weight = None, None
        else:
            with ctx:
                h_backbone = forward_range(model, embed(model, raw[:, :config.block_size]), 0, K)
                backbone_loss, per_offset, per_weight = multi_token_loss(model, heads, h_backbone, raw, eos_id=0)
            h_prev_tmp = h_backbone.detach()
            (backbone_loss / accumulate).backward()
            h_backbone = h_prev_tmp

        h_prev = h_backbone.detach()
        hidden_history = [h_prev]
        alpha_history = [torch.tensor(1.0, device=device)]
        block_metrics = []
        num_blocks_total = len(block_layer_ranges)
        for i, (start, end) in enumerate(block_layer_ranges):
            h_in = h_prev.detach().requires_grad_(True)
            alpha_i = boost_weights.alpha[i]
            is_last = (i == num_blocks_total - 1)
            own_readout_fn = frozen_readout if (FREEZE_READOUT_MODE == "except-last" and not is_last) else None
            with ctx:
                h_i = forward_range(model, h_in, start, end)
                loss_i, diag_i = chunked_boost_newton_loss(
                    model, hidden_history, alpha_history, h_i, alpha_i, y,
                    lam=NEWTON_LAMBDA, beta=NEWTON_BETA, include_curvature=INCLUDE_CURVATURE,
                    trust_region=TRUST_REGION, own_readout_fn=own_readout_fn)
            (loss_i / accumulate).backward()
            if collect_metrics:
                with torch.no_grad():
                    h_i_det = h_i.detach()
                    cka_val = linear_cka(h_prev.reshape(-1, h_prev.size(-1)),
                                          h_i_det.reshape(-1, h_i_det.size(-1))).item()
                block_metrics.append((diag_i["ce"].item(), diag_i["quad"].item(), diag_i["norm_delta_z"].item(),
                                       diag_i["descent_frac"].item(), diag_i["fix_minus_break"].item(),
                                       alpha_i.item(), cka_val))
            h_prev = h_i.detach()
            hidden_history.append(h_prev)
            alpha_history.append(alpha_i.detach())

        return per_offset, per_weight, block_metrics

    def estimate_val_loss():
        model.eval()
        generator = torch.Generator().manual_seed(eval_seed)
        num_blocks_total = len(block_layer_ranges)
        per_offset_sums = {o: 0.0 for o in (1, 2, 4)} if not FREEZE_BACKBONE else None
        block_ce_sums = [0.0] * num_blocks_total
        block_quad_sums = [0.0] * num_blocks_total
        block_norm_delta_sums = [0.0] * num_blocks_total
        block_descent_sums = [0.0] * num_blocks_total
        block_fix_minus_break_sums = [0.0] * num_blocks_total
        block_cka_sums = [0.0] * num_blocks_total
        with torch.no_grad():
            for _ in range(validation_batches):
                raw = get_raw_window("validation", micro_batch_size, config.block_size + MAX_OFFSET, device,
                                      data_cfg["data_dir"], generator=generator)
                y = raw[:, 1:1 + config.block_size]
                with ctx:
                    h_backbone = forward_range(model, embed(model, raw[:, :config.block_size]), 0, K)
                    if not FREEZE_BACKBONE:
                        _, per_offset, _ = multi_token_loss(model, heads, h_backbone, raw, eos_id=0)
                        for o, v in per_offset.items():
                            per_offset_sums[o] += v
                h_prev = h_backbone
                hidden_history = [h_prev]
                alpha_history = [torch.tensor(1.0, device=device)]
                for i, (start, end) in enumerate(block_layer_ranges):
                    alpha_i = boost_weights.alpha[i].detach()
                    with ctx:
                        h_i = forward_range(model, h_prev, start, end)
                        _, diag_i = chunked_boost_newton_loss(
                            model, hidden_history, alpha_history, h_i, alpha_i, y,
                            lam=NEWTON_LAMBDA, beta=NEWTON_BETA, include_curvature=INCLUDE_CURVATURE,
                            trust_region=TRUST_REGION)
                    cka_val = linear_cka(h_prev.reshape(-1, h_prev.size(-1)), h_i.reshape(-1, h_i.size(-1))).item()
                    block_ce_sums[i] += diag_i["ce"].item()
                    block_quad_sums[i] += diag_i["quad"].item()
                    block_norm_delta_sums[i] += diag_i["norm_delta_z"].item()
                    block_descent_sums[i] += diag_i["descent_frac"].item()
                    block_fix_minus_break_sums[i] += diag_i["fix_minus_break"].item()
                    block_cka_sums[i] += cka_val
                    h_prev = h_i
                    hidden_history.append(h_prev)
                    alpha_history.append(alpha_i)
        model.train()
        n = validation_batches
        backbone_per_offset = {o: s / n for o, s in per_offset_sums.items()} if per_offset_sums is not None else None
        return (backbone_per_offset,
                [s / n for s in block_ce_sums], [s / n for s in block_quad_sums],
                [s / n for s in block_norm_delta_sums], [s / n for s in block_descent_sums],
                [s / n for s in block_fix_minus_break_sums], [s / n for s in block_cka_sums])

    tokens_processed = 0
    for step in range(1, total_optimizer_steps + 1):
        t0 = time.perf_counter()
        lr = get_lr(step - 1, peak_lr, min_lr, warmup_steps, total_optimizer_steps)
        set_lr([optimizers["backbone"], optimizers["readout"]] + optimizers["blocks"], lr)

        collect = (step == 1 or step % train_metrics_every == 0)
        last_per_offset = last_per_weight = last_block_metrics = None
        for _ in range(grad_accum):
            last_per_offset, last_per_weight, bm = run_microbatch(grad_accum, collect)
            if collect:
                last_block_metrics = bm
            tokens_processed += micro_batch_size * config.block_size

        for group in optimizers["readout"].param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad /= num_active_readout_losses

        all_trainable = list(model.parameters()) + list(boost_weights.parameters())
        if heads is not None:
            all_trainable += list(heads.parameters())
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
                "train/lr": lr, "train/tokens_processed": tokens_processed,
                "train/tokens_per_sec": tokens_per_step / step_time, "train/step_time_s": step_time,
            }
            if last_per_offset is not None:
                for o in (1, 2, 4):
                    metrics[f"train/backbone_loss_offset{o}"] = last_per_offset[o]
                    metrics[f"train/backbone_weight_offset{o}"] = last_per_weight[o]
            for i, (start, end) in enumerate(block_layer_ranges):
                label = block_label(start, end)
                ce, quad, norm_delta, descent_frac, fix_minus_break, alpha, cka = last_block_metrics[i]
                metrics[f"train/block{label}_ce"] = ce
                metrics[f"train/block{label}_newton_quad"] = quad
                metrics[f"train/block{label}_norm_delta_z"] = norm_delta
                metrics[f"train/block{label}_descent_frac"] = descent_frac
                metrics[f"train/block{label}_fix_minus_break"] = fix_minus_break
                metrics[f"train/block{label}_alpha"] = alpha
                metrics[f"train/block{label}_cka"] = cka
            if device_type == "cuda":
                metrics["train/vram_allocated_mib"] = torch.cuda.memory_allocated() / (1024 ** 2)
                metrics["train/vram_reserved_mib"] = torch.cuda.memory_reserved() / (1024 ** 2)
            experiment.log_metrics(metrics, step=step)
            block_str = " ".join(
                f"b{block_label(s, e)}(ce={c:.2f},quad={q:.2f})"
                for (s, e), (c, q, *_rest) in zip(block_layer_ranges, last_block_metrics))
            print(f"step {step}: {block_str} grad_norm={grad_norm.item():.3f} "
                  f"tok/s={metrics['train/tokens_per_sec']:.0f}")

        is_quarter = step in quarter_labels
        if step % validation_every == 0 or is_quarter:
            (val_backbone_offset, val_block_ce, val_block_quad, val_block_norm_delta,
             val_block_descent, val_block_fix_minus_break, val_block_cka) = estimate_val_loss()
            val_metrics = {}
            if val_backbone_offset is not None:
                for o in (1, 2, 4):
                    val_metrics[f"val/backbone_loss_offset{o}"] = val_backbone_offset[o]
                    val_metrics[f"val/backbone_ppl_offset{o}"] = math.exp(val_backbone_offset[o])
            for i, (start, end) in enumerate(block_layer_ranges):
                label = block_label(start, end)
                val_metrics[f"val/block{label}_ce"] = val_block_ce[i]
                val_metrics[f"val/block{label}_ppl"] = math.exp(val_block_ce[i])
                val_metrics[f"val/block{label}_newton_quad"] = val_block_quad[i]
                val_metrics[f"val/block{label}_norm_delta_z"] = val_block_norm_delta[i]
                val_metrics[f"val/block{label}_descent_frac"] = val_block_descent[i]
                val_metrics[f"val/block{label}_fix_minus_break"] = val_block_fix_minus_break[i]
                val_metrics[f"val/block{label}_cka"] = val_block_cka[i]
            experiment.log_metrics(val_metrics, step=step)
            backbone_str = (f"val backbone_ppl_offset1={math.exp(val_backbone_offset[1]):.1f} | "
                            if val_backbone_offset is not None else "backbone заморожен | ")
            print(f"step {step}: {backbone_str}блоков в глубину: {len(block_layer_ranges)}")

            train_hparams = {
                "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
                "grad_clip": grad_clip, "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
                "weight_decay": weight_decay, "betas": betas,
            }
            sid_config = {"family": family, "k": K, "upper_block_size": UPPER_BLOCK_SIZE,
                          "block_layer_ranges": block_layer_ranges,
                          "source_checkpoint": None if FREEZE_BACKBONE else args.source_checkpoint,
                          "freeze_backbone_from": args.freeze_backbone_from, "backbone_frozen": FREEZE_BACKBONE,
                          "readout_frozen_mode": FREEZE_READOUT_MODE,
                          "newton_lambda": NEWTON_LAMBDA, "newton_beta": NEWTON_BETA,
                          "include_curvature": INCLUDE_CURVATURE, "trust_region": TRUST_REGION,
                          "upper_block_loss": "boosting + newton-quadratic"}
            checkpoint_bundle = HeadsAndWeights(heads, boost_weights)
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_sid_checkpoint(os.path.join(checkpoint_dir, "latest.pt"), model, checkpoint_bundle, optimizers,
                                 config, sid_config, tokens_processed, step, train_hparams)
            print(f"step {step}: checkpoint сохранён ({checkpoint_dir}/latest.pt)")
            if is_quarter:
                label = quarter_labels[step]
                save_sid_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{label}.pt"), model, checkpoint_bundle,
                                     optimizers, config, sid_config, tokens_processed, step, train_hparams)
                print(f"step {step}: контрольная точка {label} сохранена ({checkpoint_dir}/checkpoint_{label}.pt)")

    experiment.end()
    print(f"Newton-SID (k={K}, bs={UPPER_BLOCK_SIZE}, tag={TAG}) training loop завершён")


if __name__ == "__main__":
    main()
