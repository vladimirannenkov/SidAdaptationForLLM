"""Human-readable labels for the project's byte-level BPE tokens."""

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


TOKENIZER_DIR = Path(__file__).resolve().parents[2] / "data" / "wikitext" / "tokenizer"


def load_tokenizer():
    return ByteLevelBPETokenizer(
        str(TOKENIZER_DIR / "vocab.json"), str(TOKENIZER_DIR / "merges.txt")
    )


def token_label(tokenizer, token_id: int) -> str:
    label = tokenizer.id_to_token(int(token_id)) or str(token_id)
    return (label.replace("Ġ", "_").replace("\n", "\\n")
            .replace("\r", "\\r").replace("\t", "\\t"))
