"""Qwen3-Omni-30B-A3B-Instruct, stock, on Modal, prompted 请将这段粤语语音转换为纯文本。 (the card's Chinese ASR prompt
naming Cantonese). Details in modal_apps/qwen3_omni.py.
"""
from ._modal import batch_only, strip_tags

APP = "qwen3_omni"
transcribe = batch_only("qwen3-omni")
parse = strip_tags
