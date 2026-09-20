"""Разовое измерение (не юнит-тест): реальная CKA-матрица между backbone и
верхними блоками на уже обученном boosting-чекпоинте (k=6, полный 100М
прогон) — чтобы откалибровать штраф за линейность по факту, а не на глаз.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.wikitext.loader import get_raw_window
from src.model.gpt import GPT, GPTConfig
from src.sid.cka import linear_cka
from src.sid.checkpoint import load_sid_checkpoint
from src.sid.forward import embed, forward_range

CKPT = "checkpoints_sid_p_boost/k6_20260920_114738/checkpoint_100pct.pt"
DATA_DIR = "data/wikitext"  # запускать из корня репозитория
K = 6
NUM_UPPER_BLOCKS = 6
BATCH = 64
BLOCK_SIZE = 256
MAX_OFFSET = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = load_sid_checkpoint(CKPT)
config = GPTConfig(**ckpt["config"])
model = GPT(config).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

generator = torch.Generator().manual_seed(1337)
raw = get_raw_window("validation", BATCH, BLOCK_SIZE + MAX_OFFSET, device, DATA_DIR, generator=generator)

hiddens = []
with torch.no_grad():
    h = forward_range(model, embed(model, raw[:, :BLOCK_SIZE]), 0, K)
    hiddens.append(("backbone", h))
    for i, layer_idx in enumerate(range(K, config.n_layer)):
        h = forward_range(model, h, layer_idx, layer_idx + 1)
        hiddens.append((f"block{layer_idx}", h))

flat = [(name, h.reshape(-1, h.size(-1))) for name, h in hiddens]

print("CKA-матрица (backbone + верхние блоки), boosting-чекпоинт k=6, шаг 1524 (100%):\n")
names = [n for n, _ in flat]
header = "        " + " ".join(f"{n:>9s}" for n in names)
print(header)
with torch.no_grad():
    for i, (ni, hi) in enumerate(flat):
        row = []
        for j, (nj, hj) in enumerate(flat):
            row.append(linear_cka(hi, hj).item())
        print(f"{ni:>8s} " + " ".join(f"{v:9.4f}" for v in row))

print("\nCKA между СОСЕДНИМИ глубинами (то, что штрафовали бы в лоссе):")
for i in range(len(flat) - 1):
    v = linear_cka(flat[i][1], flat[i + 1][1]).item()
    print(f"  {flat[i][0]:>8s} <-> {flat[i+1][0]:<8s}: {v:.4f}")
