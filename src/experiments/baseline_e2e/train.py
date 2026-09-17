"""E2E-baseline: обычная модель, сквозной backprop по всей сети —
точка сравнения для всех SID/Newton/Cascade вариантов (docs/PLAN.md §9.2).

TOTAL_OPTIMIZER_STEPS считается из target_tokens, округлён вниз до кратного
4, чтобы 25/50/75/100% токен-бюджета попадали точно на границы optimizer
step.

Запуск (из корня репозитория):
    python src/experiments/baseline_e2e/train.py
    python src/experiments/baseline_e2e/train.py --smoke-test   # короткая проверка исправности
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/experiments/baseline_e2e/ -> repo root
from src.common.checkpoint import save_checkpoint
from src.common.comet_logger import init_experiment
from src.common.config import apply_smoke_overrides, load_config
from src.common.device import setup_device_dtype
from src.common.lr_schedule import get_lr, set_lr
from src.data.wikitext.loader import get_batch
from src.model.gpt import GPT, GPTConfig

EXPERIMENT_NAME = "baseline_e2e"
EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--smoke-test", action="store_true",
                         help="короткий прогон (мало токенов, Comet выключен, отдельный checkpoints_smoke/) "
                              "для проверки исправности после рефакторинга")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.smoke_test:
        cfg = apply_smoke_overrides(cfg, EXPERIMENT_NAME)
    m, t, data_cfg, ck = cfg["model"], cfg["train"], cfg["data"], cfg["checkpoint"]

    checkpoint_dir = str(Path(ck["dir"]))
    micro_batch_size, grad_accum = t["micro_batch_size"], t["grad_accumulation_steps"]
    grad_clip, peak_lr = t["grad_clip"], t["peak_lr"]
    min_lr = peak_lr / 10
    weight_decay, betas = t["weight_decay"], tuple(t["betas"])
    train_metrics_every, validation_every = t["train_metrics_every"], t["validation_every"]
    validation_batches, eval_seed = t["validation_batches"], t["eval_seed"]
    target_tokens = t["target_tokens"]

    device, device_type, dtype, ptdtype, ctx = setup_device_dtype()

    config = GPTConfig(**m)
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=peak_lr, betas=betas)
    total_params = sum(p.numel() for p in model.parameters())

    tokens_per_step = micro_batch_size * grad_accum * config.block_size
    total_optimizer_steps = max(4, (target_tokens // tokens_per_step // 4) * 4)
    warmup_steps = max(1, round(0.03 * total_optimizer_steps))
    quarter_labels = {total_optimizer_steps * i // 4: f"{i * 25}pct" for i in (1, 2, 3, 4)}

    family = "E2E"
    experiment = init_experiment(
        cfg["comet"],
        name=f"{family}_{total_params / 1e6:.2f}M_d{config.n_embd}_L{config.n_layer}_H{config.n_head}",
        tags=[family, "wikitext103", f"vocab{config.vocab_size}"],
        parameters={
            "family": family, "total_params": total_params,
            "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
            "vocab_size": config.vocab_size, "block_size": config.block_size,
            "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
            "effective_batch_tokens": tokens_per_step,
            "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_optimizer_steps, "target_tokens": target_tokens,
            "weight_decay": weight_decay, "grad_clip": grad_clip, "dtype": dtype,
        },
    )
    print(f"план прогона: {total_optimizer_steps} optimizer steps "
          f"({total_optimizer_steps * tokens_per_step:,} токенов), warmup={warmup_steps}, "
          f"четверти на шагах {sorted(quarter_labels)}")

    def estimate_val_loss():
        # Свой generator с фиксированным seed -> каждый вызов видит ровно те
        # же окна validation.bin, изменение val/loss отражает реальный
        # прогресс модели, а не другую случайную выборку.
        model.eval()
        generator = torch.Generator().manual_seed(eval_seed)
        losses = []
        with torch.no_grad():
            for _ in range(validation_batches):
                idx, targets = get_batch("validation", micro_batch_size, config.block_size, device,
                                          data_cfg["data_dir"], generator=generator)
                with ctx:
                    _, loss = model(idx, targets)
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    tokens_processed = 0
    for step in range(1, total_optimizer_steps + 1):
        t0 = time.perf_counter()

        lr = get_lr(step - 1, peak_lr, min_lr, warmup_steps, total_optimizer_steps)
        set_lr(optimizer, lr)

        step_losses = []
        for _ in range(grad_accum):
            idx, targets = get_batch("train", micro_batch_size, config.block_size, device, data_cfg["data_dir"])
            with ctx:
                _, loss = model(idx, targets)
            step_losses.append(loss.item())
            (loss / grad_accum).backward()
            tokens_processed += micro_batch_size * config.block_size

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device_type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - t0
        step_tokens = micro_batch_size * grad_accum * config.block_size

        if step == 1 or step % train_metrics_every == 0:
            metrics = {
                "train/loss": sum(step_losses) / len(step_losses),
                "train/grad_norm": grad_norm.item(),
                "train/clipped": float(grad_norm.item() > grad_clip),
                "train/lr": lr,
                "train/tokens_processed": tokens_processed,
                "train/tokens_per_sec": step_tokens / step_time,
                "train/step_time_s": step_time,
            }
            if device_type == "cuda":
                metrics["train/vram_allocated_mib"] = torch.cuda.memory_allocated() / (1024 ** 2)
                metrics["train/vram_reserved_mib"] = torch.cuda.memory_reserved() / (1024 ** 2)
            experiment.log_metrics(metrics, step=step)
            print(f"step {step}: loss={metrics['train/loss']:.4f} grad_norm={grad_norm.item():.3f} "
                  f"tok/s={metrics['train/tokens_per_sec']:.0f}")

        is_quarter = step in quarter_labels
        if step % validation_every == 0 or is_quarter:
            val_loss = estimate_val_loss()
            experiment.log_metrics({"val/loss": val_loss, "val/ppl": math.exp(val_loss)}, step=step)
            print(f"step {step}: val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}")

            # Checkpoint привязан к моменту валидации (та же логика, что в
            # пиннутом train.py) и гарантирует границу optimizer step. На
            # четвертях токен-бюджета дополнительно пишем именованный
            # checkpoint, который не перезатрётся следующей обычной валидацией.
            train_hparams = {
                "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
                "grad_clip": grad_clip, "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
                "weight_decay": weight_decay, "betas": betas,
            }
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(os.path.join(checkpoint_dir, "latest.pt"),
                             model, optimizer, config, tokens_processed, step, train_hparams)
            print(f"step {step}: checkpoint сохранён ({checkpoint_dir}/latest.pt)")
            if is_quarter:
                label = quarter_labels[step]
                save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{label}.pt"),
                                 model, optimizer, config, tokens_processed, step, train_hparams)
                print(f"step {step}: контрольная точка {label} токен-бюджета сохранена "
                      f"({checkpoint_dir}/checkpoint_{label}.pt)")

    experiment.end()
    print("training loop завершён, метрики отправлены в Comet")


if __name__ == "__main__":
    main()
