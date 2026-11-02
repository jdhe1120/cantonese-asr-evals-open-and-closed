"""rookes/cantocaptions-cantonese-asr, Oct 18 2026 release ("v3"; commit message "lora-v6: raw-audio share,
written-out repeats, vocal-noise clips"). Same loading as cantocaptions.py, pinned to its own commit.
"""
from ._qwen import transcribe as _transcribe
from .cantocaptions import MODEL_ID

REVISION = "aa5f1f3"  # 2026-10-18, head of main when added (2026-10-20)


def transcribe(wav_path: str) -> str:
    return _transcribe(MODEL_ID, wav_path, REVISION)
