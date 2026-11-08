"""Conformer-U2pp-Yue from the WenetSpeech-Yue paper (ASLP@NPU, 2025): a 130M-parameter Conformer (U2++, bidirectional
decoder, BPE units) trained on WenetSpeech-Yue with WeNet. WeNet worker (_wenet.py), CPU, default decode.

Weights: u2pp_conformer_yue/ of ASLP-lab/WSYue-ASR at 4a74a91 (sha256 eebf054f...), set up by scripts/fetch_models.py
into models/u2pp-yue with fbank dither turned off (training noise).
"""
from ._wenet import MODELS, transcriber

transcribe = transcriber("u2pp-yue", MODELS / "u2pp-yue")
