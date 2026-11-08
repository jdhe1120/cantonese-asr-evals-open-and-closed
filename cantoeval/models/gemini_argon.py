"""Gemini 4 Argon (Google's general multimodal model) prompted to transcribe, over generateContent.

Not a transcription model like `gemini` (3.5 Transcribe): audio goes in inline with a text prompt, default sampling.
Plumbing (keys, pacing, quotas, 429s) is _gemini_gc.py; scripts/gemini_batch.py sends the same request in batch.
First tried 2026-11-03: all 2,283 clips (53 live, 2,230 in one batch job), 63 thought tokens in total.
"""
from cantoeval.models import _gemini_gc as gc

MODEL = "gemini-4-argon"
NAME = "gemini_argon"
PROMPT = ("Transcribe this Cantonese audio verbatim as spoken Cantonese (粵語口語, e.g. 係 喺 嘅 咗 佢哋), "
          "not standard written Chinese, in traditional characters. Output only the transcript.")
QuotaExhausted, gap_s = gc.QuotaExhausted, gc.gap_s


def build_request(wav_path: str) -> dict:
    return {"contents": [{"role": "user", "parts": [{"text": PROMPT}, gc.inline_wav(wav_path)]}]}


def parse(text: str) -> str:
    return text.strip()


pace, transcribe = gc.make(__import__(__name__, fromlist=["*"]))
