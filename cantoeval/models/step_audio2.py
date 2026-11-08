"""Step-Audio-2-mini (stepfun-ai, 8.3B, Apache-2.0), stock, on Modal: the makers' ASR setup (system prompt
请记录下你所听到的语音内容。, greedy). Details in modal_apps/step_audio2.py. Its answer starts with a language tag
(<粤语>), dropped here; the raw answer stays in the cache entry.
"""
from ._modal import batch_only, strip_tags

APP = "step_audio2"
transcribe = batch_only("step-audio2")
parse = strip_tags
