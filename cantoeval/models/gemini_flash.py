"""Gemini 3.8 Flash with the same neutral prompt and request as gemini-argon (gemini_argon.PROMPT), so the two general
models are compared like for like."""
from cantoeval.models import _gemini_gc as gc
from cantoeval.models.gemini_argon import PROMPT, build_request, parse  # noqa: F401  (same request, other model)

MODEL = "gemini-3.8-flash"
NAME = "gemini_flash"
QuotaExhausted, gap_s = gc.QuotaExhausted, gc.gap_s

pace, transcribe = gc.make(__import__(__name__, fromlist=["*"]))
