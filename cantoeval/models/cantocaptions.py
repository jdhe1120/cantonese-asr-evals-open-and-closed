"""rookes/cantocaptions-cantonese-asr: Qwen3-ASR-1.7B + LoRA, published merged,
so it loads like a stock checkpoint.

Pinned to the Sep 20 2026 weights ("v2"), which every result in cache/cantocaptions/ came from. The Oct 18 release is
cantocaptions_v3.py, a separate row so the two versions are never mixed in one column.
"""
from ._qwen import transcribe as _transcribe

MODEL_ID = "rookes/cantocaptions-cantonese-asr"
REVISION = "d9e2680224f15065ebd10591ca1c8232da907c9b"  # 2026-09-20, "Update README.md" (weights from 3d7c51e)


def transcribe(wav_path: str) -> str:
    return _transcribe(MODEL_ID, wav_path, REVISION)
