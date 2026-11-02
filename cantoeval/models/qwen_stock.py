"""Stock Qwen/Qwen3-ASR-1.7B-hf baseline (downloads ~4 GB on first use)."""
from ._qwen import transcribe as _transcribe

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"  # the snapshot every cached result came from


def transcribe(wav_path: str) -> str:
    return _transcribe(MODEL_ID, wav_path, REVISION)
