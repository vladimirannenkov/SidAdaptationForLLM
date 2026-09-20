"""CPU-only linear probe on the residual stream after six frozen GPT blocks."""
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # tools/analysis/ -> tools/ -> repo root
from src.model.gpt import GPT, GPTConfig


torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
DEVICE = torch.device("cpu")
K = 6
BATCH_SIZE = 2
STEPS = 200
EVAL_BATCHES = 20
LR = 3e-3
BLOCK = GPTConfig().block_size


def batch(split, generator):
    data = np.memmap(f"data/wikitext/bin/{split}.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH_SIZE,), generator=generator)
    x = torch.stack([torch.from_numpy(data[int(i):int(i) + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[int(i) + 1:int(i) + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x, y


def hidden_at_k(model, x):
    pos = torch.arange(BLOCK, device=DEVICE)
    h = model.transformer.drop(model.transformer.wte(x) + model.transformer.wpe(pos))
    with torch.no_grad():
        for layer in model.transformer.h[:K]:
            h = layer(h)
    return h.detach()


def evaluate(model, probe, generator):
    total = 0.0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = batch("validation", generator)
            logits = probe(hidden_at_k(model, x))
            total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    return total / EVAL_BATCHES


checkpoint = torch.load("checkpoints/checkpoint_100pct.pt", map_location="cpu", weights_only=False)
model = GPT(GPTConfig()).to(DEVICE).eval()
model.load_state_dict(checkpoint["model"])
del checkpoint
gc.collect()
for parameter in model.parameters():
    parameter.requires_grad_(False)

probe = nn.Linear(GPTConfig().n_embd, GPTConfig().vocab_size, bias=False).to(DEVICE)
optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=0.01)
eval_generator = torch.Generator().manual_seed(1337)
initial_ce = evaluate(model, probe, eval_generator)
train_generator = torch.Generator().manual_seed(20260920)
print(f"initial probe CE={initial_ce:.4f} PPL={np.exp(initial_ce):.2f}", flush=True)

started = time.perf_counter()
for step in range(1, STEPS + 1):
    x, y = batch("train", train_generator)
    logits = probe(hidden_at_k(model, x))
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
    optimizer.step()
    if step == 1 or step % 50 == 0:
        print(f"step={step} train_ce={loss.item():.4f} elapsed={time.perf_counter() - started:.1f}s", flush=True)

final_generator = torch.Generator().manual_seed(1337)
final_ce = evaluate(model, probe, final_generator)
result = {
    "checkpoint": "checkpoints/checkpoint_100pct.pt",
    "frozen_blocks": K,
    "batch_size": BATCH_SIZE,
    "steps": STEPS,
    "initial_cross_entropy": initial_ce,
    "initial_ppl": float(np.exp(initial_ce)),
    "validation_cross_entropy": final_ce,
    "validation_ppl": float(np.exp(final_ce)),
    "cpu_threads": torch.get_num_threads(),
}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/linear_probe_k6_cpu.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
