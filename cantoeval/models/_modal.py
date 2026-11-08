"""Shared bits for models that run on a rented GPU through Modal (modal_apps/). They are filled in batches by
scripts/modal_batch.py, never clip by clip from the runner: a per-clip call would start a GPU and load the model each
time. So transcribe() only says how to fill the cache.
"""
import re


def batch_only(name: str):
    def transcribe(wav_path: str) -> str:
        raise RuntimeError(f"{name} runs on Modal in batches: uv run --with modal python scripts/modal_batch.py {name}")
    return transcribe


def strip_tags(text: str) -> str:
    """Drop leading language tags like <粤语> or <|yue|>."""
    return re.sub(r"^\s*(<\|?[^<>|]{1,12}\|?>\s*)+", "", text).strip()
