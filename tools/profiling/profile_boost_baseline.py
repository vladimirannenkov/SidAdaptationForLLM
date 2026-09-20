"""Baseline profiler: same shape as profile_newton_step.py but using the
ALREADY-PROVEN chunked_boost_loss (no Newton extras) at the ORIGINAL
MICRO_BATCH_SIZE=64/chunk_size=64 that train_sid_p_boost.py used successfully
yesterday (2026-09-21). Purpose: determine whether today's severe slowdown
(2026-09-22) is specific to the new Newton code, or an environmental/GPU-
contention issue affecting even the old, working mechanism (e.g. because the
desktop now has many more apps/games open sharing the same GPU)."""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.checkpoint import load_checkpoint
from src.data.wikitext.loader import get_raw_window
from src.model.gpt import GPT, GPTConfig
from src.sid.chunked import chunked_boost_loss
from src.sid.forward import embed, forward_range
from src.sid.losses import MAX_OFFSET, BoostWeights

DATA_DIR = "data/wikitext"  # запускать из корня репозитория

device = "cuda" if torch.cuda.is_available() else "cpu"
MICRO_BATCH_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 64
CHUNK_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 64
print(f"device={device} MICRO_BATCH_SIZE={MICRO_BATCH_SIZE} CHUNK_SIZE={CHUNK_SIZE}", flush=True)

config = GPTConfig()
K = 6
model = GPT(config)
source = load_checkpoint("checkpoints/checkpoint_50pct.pt")
model.load_state_dict(source["model"])
model.to(device)
for p in model.transformer.wte.parameters():
    p.requires_grad = False
for p in model.transformer.wpe.parameters():
    p.requires_grad = False
for layer in model.transformer.h[:K]:
    for p in layer.parameters():
        p.requires_grad = False

BLOCK_LAYER_RANGES = [(s, s + 1) for s in range(K, config.n_layer)]
boost_weights = BoostWeights(len(BLOCK_LAYER_RANGES)).to(device)

dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else torch.enable_grad()


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


for mb in range(3):
    raw = get_raw_window("train", MICRO_BATCH_SIZE, config.block_size + MAX_OFFSET, device, DATA_DIR)
    y = raw[:, 1:1 + config.block_size]

    sync()
    t0 = time.perf_counter()
    with torch.no_grad(), ctx:
        h_backbone = forward_range(model, embed(model, raw[:, :config.block_size]), 0, K)
    sync()
    print(f"[mb {mb}] backbone forward: {(time.perf_counter() - t0) * 1000:.1f} ms", flush=True)

    h_prev = h_backbone.detach()
    hidden_history = [h_prev]
    alpha_history = [torch.tensor(1.0, device=device)]
    for i, (start, end) in enumerate(BLOCK_LAYER_RANGES):
        t0 = time.perf_counter()
        h_in = h_prev.detach().requires_grad_(True)
        alpha_i = boost_weights.alpha[i]
        with ctx:
            h_i = forward_range(model, h_in, start, end)
        sync()
        t_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        with ctx:
            ce_i = chunked_boost_loss(model, hidden_history, alpha_history, h_i, alpha_i, y, chunk_size=CHUNK_SIZE)
        sync()
        t_loss = time.perf_counter() - t0

        t0 = time.perf_counter()
        ce_i.backward()
        sync()
        t_bwd = time.perf_counter() - t0

        print(f"[mb {mb}] block{start}: fwd={t_fwd * 1000:.1f}ms loss={t_loss * 1000:.1f}ms "
              f"bwd={t_bwd * 1000:.1f}ms hist_len={len(hidden_history)}", flush=True)

        h_prev = h_i.detach()
        hidden_history.append(h_prev)
        alpha_history.append(alpha_i.detach())

    model.zero_grad(set_to_none=True)

print("done", flush=True)
