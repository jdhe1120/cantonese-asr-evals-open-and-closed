"""SenseVoice-Small-Yue from the WenetSpeech-Yue paper (ASLP@NPU, 2025): SenseVoiceSmall (234M) fine-tuned on
WenetSpeech-Yue. Run exactly like the stock `sensevoice` row (FunASR, the same fsmn-vad, language yue, ITN on), so the
two rows differ only in the weights.

Weights: sensevoice_small_yue/ of ASLP-lab/WSYue-ASR at 4a74a91 (sha256 38940ad5...), set up by
scripts/fetch_models.py into models/sensevoice-yue.
"""
import re

from ._wenet import MODELS
from .sensevoice import VAD_REVISION

_model = None


def _load():
    global _model
    if _model is None:
        from funasr import AutoModel
        from huggingface_hub import snapshot_download

        from ._qwen import device

        path = MODELS / "sensevoice-yue"
        if not (path / "model.pt").exists():
            raise RuntimeError(f"sensevoice-yue not found in {path}: run scripts/fetch_models.py sensevoice-yue")
        _model = AutoModel(
            model=str(path),
            vad_model=snapshot_download("funasr/fsmn-vad", revision=VAD_REVISION),
            vad_kwargs={"max_single_segment_time": 30000},
            device=device(),
            hub="hf",
            disable_update=True,
        )
    return _model


def transcribe(wav_path: str) -> str:
    result = _load().generate(input=wav_path, language="yue", use_itn=True, batch_size_s=60, merge_vad=False)
    if not result:
        raise RuntimeError("SenseVoice-Yue returned no result")
    return re.sub(r"<\|[^|]*\|>", "", "".join(r["text"] for r in result)).strip()
