"""SID-F: backbone из готового E2E-чекпоинта, ЗАМОРОЖЕН; верхние слои
переинициализированы и обучаются каждый своим ЛОКАЛЬНЫМ CE (docs/PLAN.md
§5.1, §5.2, §4.3-4.4). Реализован обязательный контроль lambda=0 —
consistency-KL между соседними глубинами здесь не реализован (это
sid_s_blocks/sid_p_blocks, полная формула).

Доп. флаги:
--lambda-div: штраф за линейность loss_i += lambda_div * CKA(sg(h_prev), h_i)
  — явно поощряет блок вносить дополнительную (не линейно выводимую из
  предыдущей глубины) информацию. CKA логируется как диагностика всегда.
--freeze-readout {none,except-last,all}: во что бьёт градиент общего
  readout, который иначе усредняется по всем блокам.

Запуск (из корня репозитория, два разных логичных k — см. docs/starts.md):
    python src/experiments/sid_f/train.py --k 3
    python src/experiments/sid_f/train.py --k 5
    python src/experiments/sid_f/train.py --smoke-test
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/experiments/sid_f/ -> repo root
from src.common.checkpoint import load_checkpoint
from src.common.comet_logger import init_experiment
from src.common.config import apply_smoke_overrides, load_config
from src.common.device import setup_device_dtype
from src.common.lr_schedule import get_lr, set_lr
from src.data.wikitext.loader import get_batch
from src.model.gpt import GPT, GPTConfig
from src.sid.checkpoint import save_sid_checkpoint
from src.sid.cka import linear_cka
from src.sid.forward import embed, forward_range, frozen_readout, readout
from src.sid.optimizers import configure_sid_optimizers

EXPERIMENT_NAME = "sid_f"
EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args(sid_cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--k", type=int, default=sid_cfg["k"], help="глубина backbone (граница SID)")
    parser.add_argument("--source-checkpoint", default=sid_cfg["source_checkpoint"])
    parser.add_argument("--target-tokens", type=int, default=None,
                         help="переопределяет train.target_tokens из config.yaml")
    parser.add_argument("--block-size", type=int, default=sid_cfg["block_size"])
    parser.add_argument("--lambda-div", type=float, default=sid_cfg["lambda_div"])
    parser.add_argument("--freeze-readout", choices=["none", "except-last", "all"], default=sid_cfg["freeze_readout"])
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
    LAMBDA_DIV = args.lambda_div
    FREEZE_READOUT_MODE = args.freeze_readout

    micro_batch_size, grad_accum = t["micro_batch_size"], t["grad_accumulation_steps"]
    grad_clip, peak_lr = t["grad_clip"], t["peak_lr"]
    min_lr = peak_lr / 10
    weight_decay, betas = t["weight_decay"], tuple(t["betas"])
    train_metrics_every, validation_every = t["train_metrics_every"], t["validation_every"]
    validation_batches, eval_seed = t["validation_batches"], t["eval_seed"]
    target_tokens = args.target_tokens if args.target_tokens is not None else t["target_tokens"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_parts = []
    if LAMBDA_DIV > 0:
        tag_parts.append(f"div{LAMBDA_DIV:g}")
    if FREEZE_READOUT_MODE != "none":
        tag_parts.append(f"frzreadout-{FREEZE_READOUT_MODE}")
    TAG = "_".join(tag_parts) if tag_parts else "base"
    checkpoint_dir = os.path.join(ck["dir"], f"k{K}_bs{UPPER_BLOCK_SIZE}_{TAG}_{run_id}")

    device, device_type, dtype, ptdtype, ctx = setup_device_dtype()

    config = GPTConfig(**m)
    assert 0 < K < config.n_layer, f"k должен быть строго между 0 и n_layer={config.n_layer} (получено {K})"
    model = GPT(config)

    # 1. Backbone — из готового E2E-чекпоинта (docs/PLAN.md §9.2 п.3), не со
    # случайных весов.
    source = load_checkpoint(args.source_checkpoint)
    model.load_state_dict(source["model"])
    model.to(device)
    print(f"backbone загружен из {args.source_checkpoint} "
          f"(шаг {source['optimizer_step']}, {source['tokens_processed']:,} токенов)")

    # 2. Заморозить backbone (embeddings + h[0:K]).
    for p in model.transformer.wte.parameters():
        p.requires_grad = False
    for p in model.transformer.wpe.parameters():
        p.requires_grad = False
    for layer in model.transformer.h[:K]:
        for p in layer.parameters():
            p.requires_grad = False

    # 2b. --freeze-readout all: readout тоже заморожен, наравне с backbone.
    if FREEZE_READOUT_MODE == "all":
        for p in model.transformer.ln_f.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False

    # 3. Переинициализировать верхние слои — та же схема, что в GPT.__init__.
    for layer in model.transformer.h[K:]:
        layer.apply(model._init_weights)
        for pn, p in layer.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    # 4. Optimizer'ы: backbone-группа получится пустой (все requires_grad=False) — не используем её.
    optimizers = configure_sid_optimizers(model, K, weight_decay=weight_decay, learning_rate=peak_lr, betas=betas,
                                           upper_block_size=UPPER_BLOCK_SIZE)
    num_blocks = len(optimizers["blocks"])
    assert num_blocks == -(-(config.n_layer - K) // UPPER_BLOCK_SIZE)

    block_layer_ranges = [(s, min(s + UPPER_BLOCK_SIZE, config.n_layer))
                           for s in range(K, config.n_layer, UPPER_BLOCK_SIZE)]
    assert len(block_layer_ranges) == num_blocks

    def block_label(start, end):
        return str(start) if end - start == 1 else f"{start}to{end - 1}"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokens_per_step = micro_batch_size * grad_accum * config.block_size
    total_optimizer_steps = max(4, (target_tokens // tokens_per_step // 4) * 4)
    warmup_steps = max(1, round(0.03 * total_optimizer_steps))
    quarter_labels = {total_optimizer_steps * i // 4: f"{i * 25}pct" for i in (1, 2, 3, 4)}

    family = "SID-F"
    tags = [family, "wikitext103", f"vocab{config.vocab_size}", f"k{K}", f"blocksize{UPPER_BLOCK_SIZE}",
            "local-ce-only", "lambda0"]
    if LAMBDA_DIV > 0:
        tags.append(f"lambda-div{LAMBDA_DIV:g}")
    if FREEZE_READOUT_MODE != "none":
        tags.append(f"freeze-readout-{FREEZE_READOUT_MODE}")
    experiment = init_experiment(
        cfg["comet"],
        name=f"{family}_{total_params / 1e6:.2f}M_k{K}_bs{UPPER_BLOCK_SIZE}_lambda0_{TAG}_{run_id}",
        tags=tags,
        parameters={
            "run_id": run_id, "family": family, "k": K, "num_blocks": num_blocks, "upper_block_size": UPPER_BLOCK_SIZE,
            "source_checkpoint": args.source_checkpoint, "lambda_div": LAMBDA_DIV,
            "readout_frozen_mode": FREEZE_READOUT_MODE,
            "total_params": total_params, "trainable_params": trainable_params,
            "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
            "vocab_size": config.vocab_size, "block_size": config.block_size,
            "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
            "effective_batch_tokens": tokens_per_step,
            "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_optimizer_steps, "target_tokens": target_tokens,
            "weight_decay": weight_decay, "grad_clip": grad_clip, "dtype": dtype, "consistency_lambda": 0.0,
        },
    )
    print(f"k={K}: {num_blocks} верхних блоков, {trainable_params:,} обучаемых из {total_params:,} параметров")
    print(f"план прогона: {total_optimizer_steps} optimizer steps "
          f"({total_optimizer_steps * tokens_per_step:,} токенов), warmup={warmup_steps}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_file = open(os.path.join(checkpoint_dir, "metrics.jsonl"), "a", encoding="utf-8")

    def local_block_step(idx, targets, accumulate):
        """h_i = F_i(sg(h_{i-1})): backbone под no_grad (заморожен), затем
        после каждого блока h.detach() — градиент блока i+1 никогда не
        доходит до блока i."""
        with torch.no_grad(), ctx:
            h = forward_range(model, embed(model, idx), 0, K)

        per_block_loss = []
        per_block_cka = []
        num_blocks_total = len(block_layer_ranges)
        h_prev = h.detach()
        for i, (start, end) in enumerate(block_layer_ranges):
            h_in = h_prev.detach().requires_grad_(True)
            is_last = (i == num_blocks_total - 1)
            own_readout = frozen_readout if (FREEZE_READOUT_MODE == "except-last" and not is_last) else readout
            with ctx:
                h_next = forward_range(model, h_in, start, end)
                logits_i = own_readout(model, h_next)
                ce_i = F.cross_entropy(logits_i.reshape(-1, logits_i.size(-1)), targets.reshape(-1))
                if LAMBDA_DIV > 0:
                    div_i = linear_cka(h_prev.reshape(-1, h_prev.size(-1)), h_next.reshape(-1, h_next.size(-1)))
                    total_i = ce_i + LAMBDA_DIV * div_i
                else:
                    total_i = ce_i
            (total_i / accumulate).backward()
            per_block_loss.append(ce_i.item())
            with torch.no_grad():
                h_next_det = h_next.detach()
                per_block_cka.append(linear_cka(h_prev.reshape(-1, h_prev.size(-1)),
                                                 h_next_det.reshape(-1, h_next_det.size(-1))).item())
            h_prev = h_next_det
        return per_block_loss, per_block_cka

    def estimate_val_loss():
        model.eval()
        generator = torch.Generator().manual_seed(eval_seed)
        sums = [0.0] * num_blocks
        cka_sums = [0.0] * num_blocks
        with torch.no_grad():
            for _ in range(validation_batches):
                idx, targets = get_batch("validation", micro_batch_size, config.block_size, device,
                                          data_cfg["data_dir"], generator=generator)
                h = forward_range(model, embed(model, idx), 0, K)
                for i, (start, end) in enumerate(block_layer_ranges):
                    h_next = forward_range(model, h, start, end)
                    logits_i = readout(model, h_next)
                    loss_i = F.cross_entropy(logits_i.reshape(-1, logits_i.size(-1)), targets.reshape(-1))
                    sums[i] += loss_i.item()
                    cka_sums[i] += linear_cka(h.reshape(-1, h.size(-1)), h_next.reshape(-1, h_next.size(-1))).item()
                    h = h_next
        model.train()
        return [s / validation_batches for s in sums], [s / validation_batches for s in cka_sums]

    tokens_processed = 0
    for step in range(1, total_optimizer_steps + 1):
        t0 = time.perf_counter()
        lr = get_lr(step - 1, peak_lr, min_lr, warmup_steps, total_optimizer_steps)
        set_lr([optimizers["readout"]] + optimizers["blocks"], lr)

        step_block_losses = [[] for _ in range(num_blocks)]
        step_block_ckas = [[] for _ in range(num_blocks)]
        for _ in range(grad_accum):
            idx, targets = get_batch("train", micro_batch_size, config.block_size, device, data_cfg["data_dir"])
            losses, ckas = local_block_step(idx, targets, grad_accum)
            for i, (l, c) in enumerate(zip(losses, ckas)):
                step_block_losses[i].append(l)
                step_block_ckas[i].append(c)
            tokens_processed += micro_batch_size * config.block_size

        # Общий readout используется каждым блоком (или только последним при
        # --freeze-readout except-last) -> его накопленный градиент
        # усредняется по числу активных локальных лоссов.
        num_active_readout_losses = 1 if FREEZE_READOUT_MODE == "except-last" else num_blocks
        for group in optimizers["readout"].param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad /= num_active_readout_losses

        all_trainable = list(model.transformer.ln_f.parameters()) + list(model.lm_head.parameters())
        for layer in model.transformer.h[K:]:
            all_trainable += list(layer.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, grad_clip)

        for opt in [optimizers["readout"]] + optimizers["blocks"]:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if device_type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - t0

        if step == 1 or step % train_metrics_every == 0:
            metrics = {
                "train/grad_norm": grad_norm.item(), "train/clipped": float(grad_norm.item() > grad_clip),
                "train/lr": lr, "train/tokens_processed": tokens_processed,
                "train/tokens_per_sec": tokens_per_step / step_time, "train/step_time_s": step_time,
            }
            for i, (start, end) in enumerate(block_layer_ranges):
                metrics[f"train/loss_block{block_label(start, end)}"] = sum(step_block_losses[i]) / len(step_block_losses[i])
                metrics[f"train/cka_block{block_label(start, end)}"] = sum(step_block_ckas[i]) / len(step_block_ckas[i])
            if device_type == "cuda":
                metrics["train/vram_allocated_mib"] = torch.cuda.memory_allocated() / (1024 ** 2)
                metrics["train/vram_reserved_mib"] = torch.cuda.memory_reserved() / (1024 ** 2)
            experiment.log_metrics(metrics, step=step)
            metrics_file.write(json.dumps({"step": step, **metrics}) + "\n")
            metrics_file.flush()
            loss_str = " ".join(f"b{block_label(s, e)}={metrics[f'train/loss_block{block_label(s, e)}']:.3f}"
                                 for s, e in block_layer_ranges)
            print(f"step {step}: {loss_str} grad_norm={grad_norm.item():.3f} tok/s={metrics['train/tokens_per_sec']:.0f}")

        is_quarter = step in quarter_labels
        if step % validation_every == 0 or is_quarter:
            val_losses, val_ckas = estimate_val_loss()
            val_metrics = {}
            for i, (start, end) in enumerate(block_layer_ranges):
                val_metrics[f"val/loss_block{block_label(start, end)}"] = val_losses[i]
                val_metrics[f"val/ppl_block{block_label(start, end)}"] = math.exp(val_losses[i])
                val_metrics[f"val/cka_block{block_label(start, end)}"] = val_ckas[i]
            experiment.log_metrics(val_metrics, step=step)
            metrics_file.write(json.dumps({"step": step, **val_metrics}) + "\n")
            metrics_file.flush()
            print(f"step {step}: val " + " ".join(f"b{block_label(s, e)}_ppl={math.exp(val_losses[i]):.2f}"
                                                    for i, (s, e) in enumerate(block_layer_ranges)))

            train_hparams = {
                "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
                "grad_clip": grad_clip, "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
                "weight_decay": weight_decay, "betas": betas,
            }
            sid_config = {"family": family, "k": K, "consistency_lambda": 0.0,
                          "source_checkpoint": args.source_checkpoint, "upper_block_size": UPPER_BLOCK_SIZE,
                          "lambda_div": LAMBDA_DIV, "readout_frozen_mode": FREEZE_READOUT_MODE}
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_sid_checkpoint(os.path.join(checkpoint_dir, "latest.pt"), model, None, optimizers,
                                 config, sid_config, tokens_processed, step, train_hparams)
            print(f"step {step}: checkpoint сохранён ({checkpoint_dir}/latest.pt)")
            if is_quarter:
                label = quarter_labels[step]
                save_sid_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{label}.pt"), model, None, optimizers,
                                     config, sid_config, tokens_processed, step, train_hparams)
                print(f"step {step}: контрольная точка {label} сохранена ({checkpoint_dir}/checkpoint_{label}.pt)")

    experiment.end()
    metrics_file.close()
    print(f"SID-F (k={K}, lambda=0) training loop завершён")


if __name__ == "__main__":
    main()
