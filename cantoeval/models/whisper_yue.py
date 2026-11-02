"""JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english, a Cantonese-English Whisper fine-tune.

Decode settings: language yue, no_repeat_ngram_size 4, repetition_penalty 1.3. Whisper's output is often Simplified/mixed script; the scorer's t2s covers that.
"""
_pipe = None

GENERATE_KWARGS = {"language": "yue", "task": "transcribe", "no_repeat_ngram_size": 4, "repetition_penalty": 1.3}
MODEL_ID = "JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english"
REVISION = "58d693f33007f3b44b4657d7c063523c01574918"  # the snapshot every cached result came from


def transcribe(wav_path: str) -> str:
    global _pipe
    if _pipe is None:
        from transformers import pipeline

        from ._qwen import device

        _pipe = pipeline("automatic-speech-recognition", model=MODEL_ID, revision=REVISION, device=device())
    import soundfile as sf

    # Whisper refuses >30 s input unless it predicts timestamps (long-form decoding); only the whole-story clips hit this.
    long = {"return_timestamps": True} if sf.info(wav_path).duration > 30 else {}
    return _pipe(wav_path, generate_kwargs=GENERATE_KWARGS, **long)["text"].strip()
