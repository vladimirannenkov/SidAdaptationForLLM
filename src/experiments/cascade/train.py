"""Cascade-SID: sequential frozen-prefix training. Embeddings+readout are
loaded from a finished E2E checkpoint and stay frozen for the whole run;
upper layers (grouped into blocks of --block-size layers each) train ONE
BLOCK AT A TIME (each stage completed and frozen before the next starts),
each stage's own "innovation" (block output minus block input) scaled by a
learnable alpha and added to the frozen prefix's logits. Loss is three-zone
(src/cascade/losses.py::cascade_three_zone_loss): tokens where the frozen
prefix is already confidently wrong/right/uncertain get different treatment
(correct/refine/preserve), so a new stage focuses on what previous stages
have NOT already solved rather than duplicating their work.

--k — глубина замороженного (не переинициализируемого) backbone из
source-checkpoint; k=0 означает "без backbone" — все n_layer слоёв
переинициализированы и обучаются каскадом, из чекпоинта берутся только
embeddings и readout (они всё равно всегда заморожены, независимо от k).
--block-size — сколько слоёв объединяются в один stage (по умолчанию 1,
как в первой версии).

Запуск (из корня репозитория):
    python src/experiments/cascade/train.py --k 6 --block-size 1
    python src/experiments/cascade/train.py --k 6 --block-size 2
    python src/experiments/cascade/train.py --k 0 --block-size 3
    python src/experiments/cascade/train.py --smoke-test
    python src/experiments/cascade/train.py --resume checkpoints_cascade/k6_bs1_.../latest.pt
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/experiments/cascade/ -> repo root
from src.common.checkpoint import load_checkpoint as load_e2e_checkpoint
from src.common.comet_logger import init_experiment
from src.common.config import apply_smoke_overrides, load_config
from src.common.device import setup_device_dtype
from src.common.lr_schedule import get_lr, set_lr
from src.data.wikitext.loader import get_batch
from src.model.gpt import GPT, GPTConfig
from src.sid.cka import linear_cka
from src.sid.forward import embed, forward_range
from src.cascade.checkpoint import load_checkpoint, save_checkpoint
from src.cascade.losses import cascade_three_zone_loss

EXPERIMENT_NAME = "cascade"
EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args(cascade_cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--k", type=int, default=cascade_cfg["k"])
    parser.add_argument("--block-size", type=int, default=cascade_cfg.get("block_size", 1),
                         help="сколько слоёв объединяются в один stage (по умолчанию 1)")
    parser.add_argument("--source-checkpoint", default=cascade_cfg["source_checkpoint"])
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Cascade latest.pt to continue exactly")
    parser.add_argument("--correct-threshold", type=float, default=cascade_cfg["correct_threshold"])
    parser.add_argument("--preserve-threshold", type=float, default=cascade_cfg["preserve_threshold"])
    parser.add_argument("--refine-weight", type=float, default=cascade_cfg["refine_weight"])
    parser.add_argument("--preserve-weight", type=float, default=cascade_cfg["preserve_weight"])
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--grad-accumulation-steps", type=int, default=None)
    return parser.parse_args()


def main():
    cfg = load_config(EXPERIMENT_DIR / "config.yaml")
    args = parse_args(cfg["cascade"])
    cfg = load_config(args.config)
    if args.smoke_test:
        cfg = apply_smoke_overrides(cfg, EXPERIMENT_NAME)
    m, t, data_cfg, ck = cfg["model"], cfg["train"], cfg["data"], cfg["checkpoint"]

    micro_batch_size = args.micro_batch_size if args.micro_batch_size is not None else t["micro_batch_size"]
    grad_accum = args.grad_accumulation_steps if args.grad_accumulation_steps is not None else t["grad_accumulation_steps"]
    grad_clip, peak_lr = t["grad_clip"], t["peak_lr"]
    weight_decay, betas = t["weight_decay"], tuple(t["betas"])
    validation_batches, eval_seed = t["validation_batches"], t["eval_seed"]
    target_tokens = args.target_tokens if args.target_tokens is not None else t["target_tokens"]

    device, device_type, dtype, ptdtype, ctx = setup_device_dtype()

    resume_state = load_checkpoint(args.resume) if args.resume else None
    if resume_state:
        # A resumed run is defined by its original experiment, not new defaults.
        saved = resume_state["config"]
        args.k = saved["k"]
        args.block_size = saved.get("upper_block_size", 1)
        args.correct_threshold = saved["correct_threshold"]
        args.preserve_threshold = saved["preserve_threshold"]
        args.refine_weight = saved["refine_weight"]
        args.preserve_weight = saved["preserve_weight"]
        target_tokens = saved["target_tokens"]

    config = GPTConfig(**m)
    K = args.k
    UPPER_BLOCK_SIZE = args.block_size
    assert 0 <= K < config.n_layer, f"k должен быть между 0 и n_layer={config.n_layer} (получено {K})"
    BLOCK_LAYER_RANGES = [(s, min(s + UPPER_BLOCK_SIZE, config.n_layer))
                          for s in range(K, config.n_layer, UPPER_BLOCK_SIZE)]
    num_stages = len(BLOCK_LAYER_RANGES)
    model = GPT(config)
    if resume_state:
        model.load_state_dict(resume_state["model"])
    else:
        source = load_e2e_checkpoint(args.source_checkpoint)
        model.load_state_dict(source["model"])
        for layer in model.transformer.h[K:]:
            layer.apply(model._init_weights)
            for name, parameter in layer.named_parameters():
                if name.endswith("c_proj.weight"):
                    torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
    model.to(device)

    # The E2E backbone/readout and every non-active stage are immutable by design.
    for parameter in model.parameters():
        parameter.requires_grad = False
    alphas = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.01, device=device))
                                     for _ in range(num_stages)])
    if resume_state:
        for alpha, value in zip(alphas, resume_state["alphas"].to(device)):
            alpha.data.copy_(value)

    tokens_per_step = micro_batch_size * grad_accum * config.block_size
    stage_steps = max(1, (target_tokens // tokens_per_step) // num_stages)
    run_id = resume_state["config"]["run_id"] if resume_state else datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.dirname(os.path.abspath(args.resume)) if args.resume else \
        os.path.join(ck["dir"], f"k{K}_bs{UPPER_BLOCK_SIZE}_{run_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_stage = resume_state["stage"] if resume_state else 0
    start_step = resume_state["stage_step"] if resume_state else 0
    tokens_processed = resume_state["tokens_processed"] if resume_state else 0

    def stage_label(start, end):
        return str(start) if end - start == 1 else f"{start}to{end - 1}"

    run_config = {
        "run_id": run_id, "k": K, "upper_block_size": UPPER_BLOCK_SIZE,
        "block_layer_ranges": BLOCK_LAYER_RANGES, "target_tokens": target_tokens,
        "correct_threshold": args.correct_threshold, "preserve_threshold": args.preserve_threshold,
        "refine_weight": args.refine_weight, "preserve_weight": args.preserve_weight,
        "stage_steps": stage_steps, "source_checkpoint": args.source_checkpoint,
    }
    experiment = init_experiment(
        cfg["comet"],
        name=f"Cascade-SID_k{K}_bs{UPPER_BLOCK_SIZE}_{run_id}",
        tags=["cascade-sid", "frozen-backbone" if K > 0 else "no-backbone", "three-zone",
              f"k{K}", f"blocksize{UPPER_BLOCK_SIZE}"],
        parameters={**run_config, "micro_batch_size": micro_batch_size,
                    "grad_accumulation_steps": grad_accum, "dtype": dtype},
    )

    def stage_optimizer(stage):
        start, end = BLOCK_LAYER_RANGES[stage]
        for index, (s, e) in enumerate(BLOCK_LAYER_RANGES):
            for layer in model.transformer.h[s:e]:
                for parameter in layer.parameters():
                    parameter.requires_grad = index == stage
        params = []
        for layer in model.transformer.h[start:end]:
            params += list(layer.parameters())
        params += [alphas[stage]]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=peak_lr, betas=betas, foreach=False)

    def frozen_prefix(x, stage):
        """Return h0 and innovations from the completed prefix without a graph."""
        with torch.no_grad(), ctx:
            h0 = forward_range(model, embed(model, x), 0, K)
            innovations = []
            previous = None
            for previous_stage in range(stage):
                s, e = BLOCK_LAYER_RANGES[previous_stage]
                inp = h0 if previous is None else h0 + previous
                out = forward_range(model, inp, s, e)
                previous = out - inp
                innovations.append(previous)
        return h0, innovations

    def evaluate(stage):
        model.eval()
        generator = torch.Generator().manual_seed(eval_seed)
        totals = {"combined_ce": 0.0, "correct_ce": 0.0, "refine_ce": 0.0, "preserve_kl": 0.0,
                  "correct_fraction": 0.0, "refine_fraction": 0.0, "preserve_fraction": 0.0,
                  "fix_minus_break": 0.0}
        with torch.no_grad():
            for _ in range(validation_batches):
                x, y = get_batch("validation", micro_batch_size, config.block_size, device,
                                  data_cfg["data_dir"], generator=generator)
                h0, previous = frozen_prefix(x, stage)
                inp = h0 if not previous else h0 + previous[-1]
                start, end = BLOCK_LAYER_RANGES[stage]
                with ctx:
                    out = forward_range(model, inp, start, end)
                    innovation = out - inp
                    _, values = cascade_three_zone_loss(model, h0, previous, alphas[:stage], innovation, alphas[stage], y,
                                                         args.correct_threshold, args.preserve_threshold,
                                                         args.refine_weight, args.preserve_weight)
                for key in totals:
                    totals[key] += values[key]
        model.train()
        return {key: value / validation_batches for key, value in totals.items()}

    for stage in range(start_stage, num_stages):
        step0 = start_step if stage == start_stage else 0
        if step0 >= stage_steps:
            start_step = 0
            continue
        optimizer = stage_optimizer(stage)
        if resume_state and stage == start_stage:
            optimizer.load_state_dict(resume_state["optimizer"])
        metrics_every = max(1, stage_steps // 30)
        validation_every = max(1, stage_steps // 5)
        stage_start, stage_end = BLOCK_LAYER_RANGES[stage]
        print(f"stage {stage + 1}/{num_stages} (layers {stage_label(stage_start, stage_end)}): "
              f"{stage_steps} steps, resume from {step0}")
        for step in range(step0 + 1, stage_steps + 1):
            t0 = time.perf_counter()
            lr = get_lr(step - 1, peak_lr, peak_lr / 10, max(1, round(stage_steps * .03)), stage_steps)
            set_lr(optimizer, lr)
            collect = step == 1 or step % metrics_every == 0
            last_values = last_representation = None
            for _ in range(grad_accum):
                x, y = get_batch("train", micro_batch_size, config.block_size, device, data_cfg["data_dir"])
                h0, previous = frozen_prefix(x, stage)
                inp = h0 if not previous else h0 + previous[-1]
                start, end = BLOCK_LAYER_RANGES[stage]
                with ctx:
                    out = forward_range(model, inp, start, end)
                    innovation = out - inp
                    loss, last_values = cascade_three_zone_loss(model, h0, previous, alphas[:stage], innovation, alphas[stage], y,
                                                                 args.correct_threshold, args.preserve_threshold,
                                                                 args.refine_weight, args.preserve_weight)
                if collect:
                    with torch.no_grad():
                        base = h0.reshape(-1, h0.size(-1))
                        current = innovation.detach().reshape(-1, innovation.size(-1))
                        reference = base if not previous else previous[-1].reshape(-1, previous[-1].size(-1))
                        last_representation = {
                            "innovation_drift": (innovation.detach().norm() / (inp.norm() + 1e-8)).item(),
                            "innovation_cka_h0": linear_cka(base, current).item(),
                            "innovation_cka_previous": linear_cka(reference, current).item(),
                        }
                (loss / grad_accum).backward()
                tokens_processed += micro_batch_size * config.block_size
            grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"] + optimizer.param_groups[1]["params"], grad_clip)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if collect:
                log = {f"train/{k}": v for k, v in last_values.items()}
                log.update({f"train/{k}": v for k, v in last_representation.items()})
                log.update({"train/lr": lr, "train/grad_norm": grad_norm.item(), "train/alpha": alphas[stage].item(),
                            "train/tokens_processed": tokens_processed, "train/step_time_s": time.perf_counter() - t0})
                if device_type == "cuda":
                    log["train/vram_allocated_mib"] = torch.cuda.memory_allocated() / (1024 ** 2)
                    log["train/vram_reserved_mib"] = torch.cuda.memory_reserved() / (1024 ** 2)
                experiment.log_metrics(log, step=tokens_processed)
                print(f"stage {stage + 1} step {step}: ce={last_values['combined_ce']:.4f} alpha={alphas[stage].item():.4f}")
            if step % validation_every == 0 or step == stage_steps:
                values = evaluate(stage)
                experiment.log_metrics({f"val/stage{stage + 1}_{k}": v for k, v in values.items()} |
                                       {f"val/stage{stage + 1}_ppl": math.exp(values["combined_ce"])}, step=tokens_processed)
                save_checkpoint(os.path.join(checkpoint_dir, "latest.pt"), model, alphas, optimizer, stage, step,
                                tokens_processed, run_config)
        save_checkpoint(os.path.join(checkpoint_dir, f"stage{stage + 1}_final.pt"), model, alphas, optimizer, stage,
                        stage_steps, tokens_processed, run_config)
        print(f"stage {stage + 1} frozen: {checkpoint_dir}")
        start_step = 0

    experiment.end()
    print("Cascade-SID training completed")


if __name__ == "__main__":
    main()
