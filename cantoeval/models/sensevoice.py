"""FunAudioLLM/SenseVoiceSmall with fsmn-vad. Emits Simplified; the scorer's
t2s handles that, so the raw output is kept as the hypothesis."""
import re

# The snapshots every cached result came from. FunASR ignores model_revision with hub="hf", so they are fetched here
# and passed as local paths.
REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
VAD_REVISION = "df20e6b30c653645fa4ff125cacfcabd1020a669"
_model = None


def _load():
    global _model
    if _model is None:
        from funasr import AutoModel
        from huggingface_hub import snapshot_download

        from ._qwen import device

        _model = AutoModel(
            model=snapshot_download("FunAudioLLM/SenseVoiceSmall", revision=REVISION),
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
        raise RuntimeError("SenseVoice returned no result")
    text = "".join(r["text"] for r in result)
    return re.sub(r"<\|[^|]*\|>", "", text).strip()  # strip language/emotion/event tags
