"""GLM-ASR-Nano-2512 with a Cantonese instruction in place of the default prompt (the model has no language setting)."""
from .glm_asr import transcribe as _transcribe

PROMPT = "請將呢段粵語錄音逐字轉寫成粵語口語。"


def transcribe(wav_path: str) -> str:
    return _transcribe(wav_path, PROMPT)
