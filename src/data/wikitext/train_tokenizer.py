"""Обучает byte-level BPE токенизатор (VOCAB_SIZE=16384) на train split
WikiText-103-raw-v1 из локальных parquet-шардов (raw_cache/, см. docs/history.md
почему не через datasets.load_dataset напрямую — нестабильная сеть рвала
длинную докачку). Результат — tokenizer/{vocab.json,merges.txt}, вход для
prepare.py.

--data-dir по умолчанию "data/wikitext" — запускать из корня репозитория:
    python src/data/wikitext/train_tokenizer.py
"""

import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import ByteLevelBPETokenizer

VOCAB_SIZE = 16384


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/wikitext")
    args = parser.parse_args()
    raw_cache_dir = os.path.join(args.data_dir, "raw_cache")
    output_dir = os.path.join(args.data_dir, "tokenizer")

    shard_paths = sorted(glob.glob(os.path.join(raw_cache_dir, "train_*.parquet")))
    dataset = pq.read_table(shard_paths[0])
    for path in shard_paths[1:]:
        dataset = pa.concat_tables([dataset, pq.read_table(path)])

    tokenizer = ByteLevelBPETokenizer()
    texts = dataset.column("text").to_pylist()

    def text_iterator():
        for text in texts:
            if text:
                yield text

    tokenizer.train_from_iterator(
        text_iterator(), vocab_size=VOCAB_SIZE, min_frequency=2,
        special_tokens=["<|endoftext|>"],
    )

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)

    real_vocab_size = tokenizer.get_vocab_size()
    print(f"обучение завершено, итоговый vocab_size = {real_vocab_size}")
    print(f"сохранено в {output_dir}")

    sample = "The quick brown fox jumps over the lazy dog. Wikipedia articles often mention 1994."
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)
    print(f"пример закодирован в {len(encoded.ids)} токенов: {encoded.tokens}")
    print(f"decode(encode(sample)) == sample -> {decoded.strip() == sample.strip()}")


if __name__ == "__main__":
    main()
