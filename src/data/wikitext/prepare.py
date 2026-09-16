"""Готовит бинарные token-стримы (bin/*.bin + doc_offsets + meta.json) из
скачанных parquet-шардов WikiText-103 (raw_cache/) через уже обученный
BPE-токенизатор (tokenizer/, см. train_tokenizer.py). Документы режутся по
заголовкам первого уровня ("= Title ="), каждый документ завершается EOS —
это даёт границы, по которым multi-token loss маскирует targets за концом
документа (src/sid/losses.py).

--data-dir по умолчанию "data/wikitext" — запускать из корня репозитория:
    python src/data/wikitext/prepare.py
"""

import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import ByteLevelBPETokenizer

LEVEL1_HEADING = re.compile(r"^= ([^=].*[^=]) =$")
LENS_HOLDOUT_STRIDE = 500


def load_split_lines(raw_cache_dir, split):
    paths = sorted(glob.glob(os.path.join(raw_cache_dir, f"{split}_*.parquet")))
    table = pq.read_table(paths[0])
    for p in paths[1:]:
        table = pa.concat_tables([table, pq.read_table(p)])
    return table.column("text").to_pylist()


def split_into_documents(lines):
    documents, current = [], []
    for line in lines:
        if LEVEL1_HEADING.match(line.strip()) and current:
            documents.append("".join(current))
            current = []
        current.append(line if line else "\n")
    if current:
        documents.append("".join(current))
    return [d for d in documents if d.strip()]


def tokenize_documents(documents, tokenizer, eos_id):
    arrays, offsets = [], [0]
    for doc in documents:
        ids = tokenizer.encode(doc).ids + [eos_id]
        arrays.append(np.array(ids, dtype=np.uint16))
        offsets.append(offsets[-1] + len(ids))
    return np.concatenate(arrays), np.array(offsets, dtype=np.int64)


def save_split(out_dir, name, tokens, doc_offsets):
    os.makedirs(out_dir, exist_ok=True)
    tokens.tofile(os.path.join(out_dir, f"{name}.bin"))
    np.save(os.path.join(out_dir, f"{name}_doc_offsets.npy"), doc_offsets)
    checksum = hashlib.sha256(tokens.tobytes()).hexdigest()
    return {"tokens": int(tokens.size), "documents": int(len(doc_offsets) - 1), "sha256": checksum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/wikitext")
    args = parser.parse_args()
    raw_cache_dir = os.path.join(args.data_dir, "raw_cache")
    tokenizer_dir = os.path.join(args.data_dir, "tokenizer")
    out_dir = os.path.join(args.data_dir, "bin")

    tokenizer = ByteLevelBPETokenizer(
        os.path.join(tokenizer_dir, "vocab.json"), os.path.join(tokenizer_dir, "merges.txt")
    )
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    vocab_size = tokenizer.get_vocab_size()

    meta = {"vocab_size": vocab_size, "eos_id": eos_id, "splits": {}}

    for split in ("train", "validation", "test"):
        documents = split_into_documents(load_split_lines(raw_cache_dir, split))

        if split == "train":
            holdout_docs = documents[::LENS_HOLDOUT_STRIDE]
            documents = [d for i, d in enumerate(documents) if i % LENS_HOLDOUT_STRIDE != 0]
            tokens, offsets = tokenize_documents(holdout_docs, tokenizer, eos_id)
            meta["splits"]["train_lens_holdout"] = save_split(out_dir, "train_lens_holdout", tokens, offsets)

        tokens, offsets = tokenize_documents(documents, tokenizer, eos_id)
        meta["splits"][split] = save_split(out_dir, split, tokens, offsets)
        info = meta["splits"][split]
        print(f"{split}: {info['tokens']:,} токенов, {info['documents']} документов")

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
