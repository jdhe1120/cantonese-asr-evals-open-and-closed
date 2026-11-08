"""MiMo-V2.5-ASR (Xiaomi, 8B, MIT), stock, on Modal: asr_sft() with automatic language detection. Details in
modal_apps/mimo_asr.py.
"""
from ._modal import batch_only, strip_tags

APP = "mimo_asr"
transcribe = batch_only("mimo-asr")
parse = strip_tags
