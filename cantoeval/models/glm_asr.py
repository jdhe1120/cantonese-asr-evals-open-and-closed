"""zai-org/GLM-ASR-Nano-2512 (1.5B, Apache-2.0), run with transformers' native GlmAsr classes (~4.5 GB on first use).

Default prompt ("Transcribe the input speech."): the model has no language setting. The processor splits audio into
30 s windows and truncates past 655 s, so long whole-story clips lose their tail.
"""
import librosa
import torch

from ._qwen import device, dtype

MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
REVISION = "61ba4e0b3309b6656edea3e93e419f7bd5c61957"  # the snapshot every cached result came from
_loaded = None


def transcribe(wav_path: str, prompt: str | None = None) -> str:
    """`prompt=None` uses the processor's default instruction."""
    global _loaded
    if _loaded is None:
        from transformers import AutoProcessor, GlmAsrForConditionalGeneration

        proc = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        model = GlmAsrForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION, dtype=dtype()).to(device()).eval()
        _loaded = (model, proc)
    model, proc = _loaded
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    inputs = proc.apply_transcription_request(audio, prompt=prompt).to(model.device, model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max(200, int(len(audio) / 16000 * 8)), do_sample=False)
    return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
