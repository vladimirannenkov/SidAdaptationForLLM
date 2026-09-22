"""SID-VocabModules: тот же boosting-механизм, что SID-P-Boost
(src/experiments/sid_p_boost/train.py — кумулятивная сумма логитов под
stop-gradient, обучаемый alpha_i на вклад каждого блока), но собственная
коррекция блока i идёт не через полный shared vocab-readout, а через
маленький линейный модуль G_i (n_embd -> proj_dim) и ФИКСИРОВАННУЮ случайную
sparse-проекцию P_i (proj_dim x vocab_size, своя для каждого блока, не
обучается):

    Δz_i = G_i(h_i) @ P_i
    z_i  = sg(z_{i-1}) + alpha_i * Δz_i
    loss_i = CE(target, softmax(z_i))

Идея: раз P_i у разных блоков разные (случайные, фиксированные с самого
начала), блок i физически не может представить произвольную full-vocabulary
коррекцию — его вклад ограничен образом P_i^T. Математически невозможно,
чтобы все блоки сошлись к одной и той же корректирующей full-vocab функции
(в отличие от обычного boosting, где каждый блок теоретически мог бы
скопировать readout(h) другого блока). Backbone заморожен из E2E-чекпоинта
(как SID-F) — простейшая база для сравнения с SID-P-Boost/SID-F без лишних
переменных (multi-token loss backbone здесь не нужен).

Запуск (из корня репозитория):
    python src/experiments/sid_vocab_modules/train.py --k 6
    python src/experiments/sid_vocab_modules/train.py --smoke-test
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # src/experiments/sid_vocab_modules/ -> repo root
from src.common.checkpoint import load_checkpoint
from src.common.comet_logger import init_experiment
from src.common.config import apply_smoke_overrides, load_config
from src.common.device import setup_device_dtype
from src.common.lr_schedule import get_lr, set_lr
from src.data.wikitext.loader import get_batch
from src.model.gpt import GPT, GPTConfig
from src.sid.cka import linear_cka
from src.sid.checkpoint import save_sid_checkpoint
from src.sid.forward import embed, forward_range, readout
from src.sid.losses import BoostWeights
from src.sid.optimizers import configure_sid_optimizers

EXPERIMENT_NAME = "sid_vocab_modules"
EXPERIMENT_DIR = Path(__file__).resolve().parent


class VocabProjectionHeads(nn.Module):
    """Один маленький линейный модуль G_i (n_embd -> proj_dim, обучаемый) и
    одна фиксированная случайная sparse-проекция P_i (proj_dim x vocab_size,
    буфер, НЕ параметр) на каждый верхний блок. P_i генерируется один раз при
    создании (фиксированный projection_seed -> воспроизводимо между
    запусками), с одной и той же долей ненулевых столбцов в каждой строке
    (sparsity) и знаком +-1/sqrt(nnz) на них (сохраняет норму в среднем, как
    в классическом sparse random projection / JL-embedding)."""

    def __init__(self, n_embd, vocab_size, num_blocks, proj_dim, sparsity, seed):
        super().__init__()
        self.g = nn.ModuleList([nn.Linear(n_embd, proj_dim) for _ in range(num_blocks)])
        for layer in self.g:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

        generator = torch.Generator().manual_seed(seed)
        nnz_per_row = max(1, int(sparsity * vocab_size))
        # Масштаб нормирован по ОЖИДАЕМОМУ числу ненулевых вкладов В ОДИН
        # выходной (vocab) столбец — proj_dim*sparsity строк в среднем задевают
        # любой данный столбец, — а не по nnz строки. Строка используется в
        # "расширяющем" направлении (proj_dim -> vocab_size), поэтому нужна
        # именно эта нормировка: она даёт std(G_i(h) @ P_i) того же порядка,
        # что и обычный readout(h) (проверено численно перед реализацией) —
        # без неё коррекция блока гаснет почти до нуля (grad_norm~1e-3).
        scale = 1.0 / max(proj_dim * sparsity, 1.0) ** 0.5
        projections = []
        for _ in range(num_blocks):
            p = torch.zeros(proj_dim, vocab_size)
            for row in range(proj_dim):
                cols = torch.randperm(vocab_size, generator=generator)[:nnz_per_row]
                signs = torch.randint(0, 2, (nnz_per_row,), generator=generator).float() * 2 - 1
                p[row, cols] = signs * scale
            projections.append(p)
        self.register_buffer("P", torch.stack(projections))  # (num_blocks, proj_dim, vocab_size), не обучается

    def correction(self, block_idx, h):
        """Δz для блока block_idx: G_i(h) @ P_i, форма (..., vocab_size)."""
        return self.g[block_idx](h) @ self.P[block_idx]


def chunked_boost_projected_loss(model, hidden_history, alpha_history, h_i, alpha_i,
                                  vocab_heads, block_idx, target, chunk_size=64):
    """Как sid/chunked.py::chunked_boost_loss (кумулятивная сумма readout
    предыдущих глубин под sg + alpha_i * СВОЯ коррекция), но своя коррекция —
    не readout(h_i), а vocab_heads.correction(block_idx, h_i) (ограничена
    случайной sparse-проекцией P_i, см. VocabProjectionHeads)."""
    T = h_i.size(1)
    total_ce = h_i.new_zeros(())
    total_tokens = 0
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        target_chunk = target[:, start:end]
        n = target_chunk.numel()

        with torch.no_grad():
            cum_logits = None
            for h_prev, alpha_prev in zip(hidden_history, alpha_history):
                alpha_prev_val = alpha_prev.detach() if torch.is_tensor(alpha_prev) else alpha_prev
                logits_prev = readout(model, h_prev[:, start:end]) * alpha_prev_val
                cum_logits = logits_prev if cum_logits is None else cum_logits + logits_prev

        delta_z = vocab_heads.correction(block_idx, h_i[:, start:end]) * alpha_i
        combined = delta_z if cum_logits is None else cum_logits + delta_z

        total_ce = total_ce + F.cross_entropy(
            combined.reshape(-1, combined.size(-1)), target_chunk.reshape(-1), reduction="sum")
        total_tokens += n

    return total_ce / total_tokens


def parse_args(sid_cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--k", type=int, default=sid_cfg["k"])
    parser.add_argument("--source-checkpoint", default=sid_cfg["source_checkpoint"])
    parser.add_argument("--proj-dim", type=int, default=sid_cfg["proj_dim"])
    parser.add_argument("--sparsity", type=float, default=sid_cfg["sparsity"])
    parser.add_argument("--projection-seed", type=int, default=sid_cfg["projection_seed"])
    return parser.parse_args()


def main():
    cfg = load_config(EXPERIMENT_DIR / "config.yaml")
    args = parse_args(cfg["sid"])
    cfg = load_config(args.config)
    if args.smoke_test:
        cfg = apply_smoke_overrides(cfg, EXPERIMENT_NAME)
    m, t, data_cfg, ck = cfg["model"], cfg["train"], cfg["data"], cfg["checkpoint"]
    K = args.k

    micro_batch_size, grad_accum = t["micro_batch_size"], t["grad_accumulation_steps"]
    grad_clip, peak_lr = t["grad_clip"], t["peak_lr"]
    min_lr = peak_lr / 10
    weight_decay, betas = t["weight_decay"], tuple(t["betas"])
    min_train_metric_points = t["min_train_metric_points"]
    validation_every, validation_batches, eval_seed = t["validation_every"], t["validation_batches"], t["eval_seed"]
    target_tokens = t["target_tokens"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(ck["dir"], f"k{K}_{run_id}")

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

    optimizers = configure_sid_optimizers(model, K, weight_decay=weight_decay, learning_rate=peak_lr, betas=betas)
    num_blocks = len(optimizers["blocks"])

    vocab_heads = VocabProjectionHeads(config.n_embd, config.vocab_size, num_blocks,
                                        args.proj_dim, args.sparsity, args.projection_seed).to(device)
    boost_weights = BoostWeights(num_blocks).to(device)
    # G_i (обучаемые) и alpha_i — отдельный optimizer, не в readout (readout
    # здесь вообще не обучается своим лоссом отдельно: backbone заморожен,
    # своей "нулевой" глубины у readout нет — только через каждый блок).
    heads_optimizer = torch.optim.AdamW(
        [{"params": list(vocab_heads.g.parameters()) + list(boost_weights.parameters()), "weight_decay": 0.0}],
        lr=peak_lr, betas=betas, foreach=False,
    )
    # readout (ln_f+lm_head) используется только для читения ПРЕДЫДУЩИХ
    # (замороженных под sg) глубин — сам он не получает градиент вообще в
    # этой схеме (все корректировки блоков идут через vocab_heads, не через
    # readout), поэтому optimizers["readout"] не используется/не шагается.

    total_params = sum(p.numel() for p in model.parameters())
    aux_params = sum(p.numel() for p in vocab_heads.g.parameters())

    tokens_per_step = micro_batch_size * grad_accum * config.block_size
    total_optimizer_steps = max(4, (target_tokens // tokens_per_step // 4) * 4)
    warmup_steps = max(1, round(0.03 * total_optimizer_steps))
    quarter_labels = {total_optimizer_steps * i // 4: f"{i * 25}pct" for i in (1, 2, 3, 4)}
    train_metrics_every = max(1, total_optimizer_steps // min_train_metric_points)

    family = "SID-VocabModules"
    experiment = init_experiment(
        cfg["comet"],
        name=f"{family}_{total_params / 1e6:.2f}M_k{K}_proj{args.proj_dim}_{run_id}",
        tags=[family, "wikitext103", f"vocab{config.vocab_size}", f"k{K}", "boosting",
              "sparse-random-projection", "frozen-backbone"],
        parameters={
            "run_id": run_id, "family": family, "k": K, "num_blocks": num_blocks,
            "proj_dim": args.proj_dim, "sparsity": args.sparsity, "projection_seed": args.projection_seed,
            "source_checkpoint": args.source_checkpoint, "aux_params": aux_params,
            "total_params": total_params,
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
    print(f"k={K}, {num_blocks} верхних блоков, proj_dim={args.proj_dim}, sparsity={args.sparsity}; "
          f"план: {total_optimizer_steps} steps ({total_optimizer_steps * tokens_per_step:,} токенов)")

    def run_microbatch(accumulate, collect_metrics):
        idx, targets = get_batch("train", micro_batch_size, config.block_size, device, data_cfg["data_dir"])
        with torch.no_grad(), ctx:
            h_backbone = forward_range(model, embed(model, idx), 0, K)

        h_prev = h_backbone.detach()
        hidden_history = [h_prev]
        alpha_history = [torch.tensor(1.0, device=device)]
        block_metrics = []
        for i in range(num_blocks):
            layer = model.transformer.h[K + i]
            h_in = h_prev.detach().requires_grad_(True)
            alpha_i = boost_weights.alpha[i]
            with ctx:
                h_i = layer(h_in)
                ce_i = chunked_boost_projected_loss(model, hidden_history, alpha_history, h_i, alpha_i,
                                                     vocab_heads, i, targets)
            (ce_i / accumulate).backward()
            if collect_metrics:
                with torch.no_grad():
                    h_i_det = h_i.detach()
                    drift = (h_i_det - h_prev).norm() / (h_prev.norm() + 1e-8)
                    cka_val = linear_cka(h_prev.reshape(-1, h_prev.size(-1)),
                                          h_i_det.reshape(-1, h_i_det.size(-1))).item()
                block_metrics.append((ce_i.item(), drift.item(), alpha_i.item(), cka_val))
            h_prev = h_i.detach()
            hidden_history.append(h_prev)
            alpha_history.append(alpha_i.detach())

        return block_metrics

    def estimate_val_loss():
        model.eval()
        generator = torch.Generator().manual_seed(eval_seed)
        block_ce_sums = [0.0] * num_blocks
        block_drift_sums = [0.0] * num_blocks
        block_cka_sums = [0.0] * num_blocks
        with torch.no_grad():
            for _ in range(validation_batches):
                idx, targets = get_batch("validation", micro_batch_size, config.block_size, device,
                                          data_cfg["data_dir"], generator=generator)
                h_backbone = forward_range(model, embed(model, idx), 0, K)
                h_prev = h_backbone
                hidden_history = [h_prev]
                alpha_history = [torch.tensor(1.0, device=device)]
                for i in range(num_blocks):
                    layer = model.transformer.h[K + i]
                    alpha_i = boost_weights.alpha[i].detach()
                    with ctx:
                        h_i = layer(h_prev)
                        ce_i = chunked_boost_projected_loss(model, hidden_history, alpha_history, h_i, alpha_i,
                                                             vocab_heads, i, targets)
                    drift = (h_i - h_prev).norm() / (h_prev.norm() + 1e-8)
                    cka_val = linear_cka(h_prev.reshape(-1, h_prev.size(-1)), h_i.reshape(-1, h_i.size(-1))).item()
                    block_ce_sums[i] += ce_i.item()
                    block_drift_sums[i] += drift.item()
                    block_cka_sums[i] += cka_val
                    h_prev = h_i
                    hidden_history.append(h_prev)
                    alpha_history.append(alpha_i)
        model.train()
        n = validation_batches
        return [s / n for s in block_ce_sums], [s / n for s in block_drift_sums], [s / n for s in block_cka_sums]

    tokens_processed = 0
    for step in range(1, total_optimizer_steps + 1):
        t0 = time.perf_counter()
        lr = get_lr(step - 1, peak_lr, min_lr, warmup_steps, total_optimizer_steps)
        set_lr([heads_optimizer] + optimizers["blocks"], lr)

        collect = step == 1 or step % train_metrics_every == 0
        last_block_metrics = None
        for _ in range(grad_accum):
            bm = run_microbatch(grad_accum, collect)
            if collect:
                last_block_metrics = bm
            tokens_processed += micro_batch_size * config.block_size

        all_trainable = list(vocab_heads.g.parameters()) + list(boost_weights.parameters())
        for layer in model.transformer.h[K:]:
            all_trainable += list(layer.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, grad_clip)

        for opt in [heads_optimizer] + optimizers["blocks"]:
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
            for i in range(num_blocks):
                label = K + i
                ce, drift, alpha, cka = last_block_metrics[i]
                metrics[f"train/block{label}_ce"] = ce
                metrics[f"train/block{label}_drift"] = drift
                metrics[f"train/block{label}_alpha"] = alpha
                metrics[f"train/block{label}_cka"] = cka
            if device_type == "cuda":
                metrics["train/vram_allocated_mib"] = torch.cuda.memory_allocated() / (1024 ** 2)
                metrics["train/vram_reserved_mib"] = torch.cuda.memory_reserved() / (1024 ** 2)
            experiment.log_metrics(metrics, step=step)
            block_str = " ".join(f"b{K+i}(ce={c:.2f},alpha={a:.3f},cka={k:.3f})"
                                  for i, (c, d, a, k) in enumerate(last_block_metrics))
            print(f"step {step}: {block_str} grad_norm={grad_norm.item():.3f} "
                  f"tok/s={metrics['train/tokens_per_sec']:.0f}")

        is_quarter = step in quarter_labels
        if step % validation_every == 0 or is_quarter:
            val_ce, val_drift, val_cka = estimate_val_loss()
            val_metrics = {}
            for i in range(num_blocks):
                label = K + i
                val_metrics[f"val/block{label}_ce"] = val_ce[i]
                val_metrics[f"val/block{label}_ppl"] = math.exp(val_ce[i])
                val_metrics[f"val/block{label}_drift"] = val_drift[i]
                val_metrics[f"val/block{label}_cka"] = val_cka[i]
            experiment.log_metrics(val_metrics, step=step)
            depth_str = " ".join(f"b{K+i}:ppl={math.exp(c):.1f},cka={k:.3f}"
                                  for i, (c, k) in enumerate(zip(val_ce, val_cka)))
            print(f"step {step}: глубина (ансамбль): {depth_str}")

            train_hparams = {
                "micro_batch_size": micro_batch_size, "grad_accumulation_steps": grad_accum,
                "grad_clip": grad_clip, "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup_steps,
                "weight_decay": weight_decay, "betas": betas,
            }
            sid_config = {"family": family, "k": K, "proj_dim": args.proj_dim, "sparsity": args.sparsity,
                          "projection_seed": args.projection_seed, "source_checkpoint": args.source_checkpoint}
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_sid_checkpoint(os.path.join(checkpoint_dir, "latest.pt"), model, vocab_heads, optimizers,
                                 config, sid_config, tokens_processed, step, train_hparams)
            print(f"step {step}: checkpoint сохранён ({checkpoint_dir}/latest.pt)")
            if is_quarter:
                label = quarter_labels[step]
                save_sid_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{label}.pt"), model, vocab_heads,
                                     optimizers, config, sid_config, tokens_processed, step, train_hparams)
                print(f"step {step}: контрольная точка {label} сохранена ({checkpoint_dir}/checkpoint_{label}.pt)")

    experiment.end()
    print(f"SID-VocabModules (k={K}, proj_dim={args.proj_dim}) training loop завершён")


if __name__ == "__main__":
    main()
