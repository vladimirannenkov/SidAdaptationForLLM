"""Проверка data/wikitext/prepare.py: реальный round-trip чтения train.bin +
границ документов (данные остаются в data/wikitext/ на верхнем уровне
репозитория, а не под src/ — это артефакты, не код)."""

import hashlib
import json
import os
import sys

import numpy as np
from tokenizers import ByteLevelBPETokenizer

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN_DIR = os.path.join(REPO_ROOT, "data", "wikitext", "bin")
TOKENIZER_DIR = os.path.join(REPO_ROOT, "data", "wikitext", "tokenizer")

tokenizer = ByteLevelBPETokenizer(
    os.path.join(TOKENIZER_DIR, "vocab.json"), os.path.join(TOKENIZER_DIR, "merges.txt")
)
eos_id = tokenizer.token_to_id("<|endoftext|>")

with open(os.path.join(BIN_DIR, "meta.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)
print(f"meta: vocab_size={meta['vocab_size']}, eos_id={meta['eos_id']}")
assert meta["eos_id"] == eos_id

tokens = np.memmap(os.path.join(BIN_DIR, "train.bin"), dtype=np.uint16, mode="r")
offsets = np.load(os.path.join(BIN_DIR, "train_doc_offsets.npy"))
print(f"train.bin: {tokens.size:,} токенов (memmap, не в RAM целиком), {len(offsets) - 1:,} документов")

# 1. Последний токен каждого из первых 5 документов должен быть EOS.
for i in range(5):
    doc_tokens = tokens[offsets[i]:offsets[i + 1]]
    assert doc_tokens[-1] == eos_id, f"документ {i}: последний токен {doc_tokens[-1]} != eos_id {eos_id}"
print("первые 5 документов заканчиваются на eos_id -> OK")

# 2. dtype и диапазон id.
assert tokens.max() < meta["vocab_size"], "id токена выходит за пределы словаря"
print(f"max token id = {tokens.max()} < vocab_size={meta['vocab_size']} -> OK, dtype={tokens.dtype}")

# 3. Декодируем первый документ целиком и смотрим текст глазами.
first_doc = tokens[offsets[0]:offsets[1]].tolist()
decoded = tokenizer.decode([t for t in first_doc if t != eos_id])
print(f"первый документ ({len(first_doc)} токенов), начало текста:\n{decoded[:200]!r}")

# 4. Контрольная сумма пересчитывается и совпадает с meta.json.
recomputed = hashlib.sha256(np.asarray(tokens).tobytes()).hexdigest()
print(f"sha256 совпадает с meta.json -> {recomputed == meta['splits']['train']['sha256']}")
