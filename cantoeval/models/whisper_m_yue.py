"""Whisper-medium-Yue from the WenetSpeech-Yue paper (ASLP@NPU, 2025): Whisper medium (769M) fine-tuned on
WenetSpeech-Yue with WeNet, run in a WeNet worker (_wenet.py), CPU, attention decoding (greedy, WeNet's default).

Start prompt language "zh": train.yaml sets no language_conf, so WeNet's data pipeline tagged each training transcript
by text language ID limited to zh/en, i.e. zh for Cantonese text. WeNet's transcribe() would decode with "en" instead
(results cached before 2026-11-07 were; about 1 in 8 gzcabin clips came out as English translations).

Weights: whisper_medium_yue/ of ASLP-lab/WSYue-ASR at 4a74a91 (sha256 32b35310...), set up by
scripts/fetch_models.py into models/whisper-m-yue.

Whisper's encoder takes at most 30 s and WeNet has no long-form decoding (it fails an assertion), so a longer clip is
cut once at its quietest 0.2 s between 6 s from either end, each half transcribed and the texts joined. Only two scored
clips are over 30 s (cantomap-p217, -p229); whisper_yue.py uses Hugging Face's long-form decoding for the same case.
"""
import tempfile
from pathlib import Path

from ._wenet import MODELS, transcriber

_run = transcriber("whisper-m-yue", MODELS / "whisper-m-yue", lang="zh")
LIMIT_S = 30.0


def transcribe(wav_path: str) -> str:
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    if len(audio) <= LIMIT_S * sr:
        return _run(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win, lo, hi = int(0.2 * sr), int(6 * sr), len(audio) - int(6 * sr)
    energy = np.convolve(audio ** 2, np.ones(win), "valid")
    cut = lo + int(np.argmin(energy[lo:hi - win])) + win // 2
    parts = [audio[:cut], audio[cut:]]
    if max(len(p) for p in parts) > LIMIT_S * sr:
        raise RuntimeError(f"{wav_path}: a half is still over {LIMIT_S:.0f} s; needs more than one cut")
    with tempfile.TemporaryDirectory() as tmp:
        texts = []
        for i, part in enumerate(parts):
            path = Path(tmp) / f"part{i}.wav"
            sf.write(path, part, sr)
            texts.append(_run(str(path)))
    return "".join(texts)
