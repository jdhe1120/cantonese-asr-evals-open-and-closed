"""Shared loader for Qwen3-ASR checkpoints (merged weights) via transformers."""
from pathlib import Path

import librosa
import torch

_loaded: dict[str, tuple] = {}


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def dtype():
    d = device()
    if d == "mps":
        return torch.bfloat16
    if d == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


def transcribe(model_id: str, wav_path: str, revision: str | None = None, prompt: str | None = None,
               suppress: tuple[int, ...] | None = None) -> str:
    """revision pins a Hugging Face commit; without it the newest weights would be fetched, silently mixing versions
    under one cache (the cache is keyed on audio + reference, not on model version). prompt goes in the system slot
    (Qwen3-ASR's context field, documented for hotwords); None leaves it empty, as every cached result so far.
    suppress lists token ids decoding may never pick."""
    key = f"{model_id}@{revision}"
    if key not in _loaded:
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        proc = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, revision=revision, dtype=dtype()).to(device()).eval()
        _loaded[key] = (model, proc)
    model, proc = _loaded[key]
    audio, _ = librosa.load(str(Path(wav_path)), sr=16000, mono=True)
    inputs = proc.apply_transcription_request(audio=audio, language="Cantonese", prompt=prompt)
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max(200, int(len(audio) / 16000 * 8)), do_sample=False,
                             **({"suppress_tokens": list(suppress)} if suppress else {}))
    # transcription_only strips the "language X<asr_text>" prefix and collapses runaway repetition.
    return proc.decode(out[:, inputs.input_ids.shape[1]:], return_format="transcription_only",
                       clean_up_tokenization_spaces=False)[0].strip()
