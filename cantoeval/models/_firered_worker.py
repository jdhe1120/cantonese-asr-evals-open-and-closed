"""Stand-alone worker for FireRedASR2-AED, run in its own isolated env (fireredasr2s pins torch 2.10 / transformers
5.1 / numpy 2.4). Started by fireredasr2.py; protocol in _worker.py. argv[1] is the model dir (the Hugging Face repo).

Decoding settings are the README's AED example (beam 3, softmax smoothing 1.25, length penalty 0.6, eos penalty 1.0),
CPU (the code only knows CUDA or CPU), no timestamps.
"""
import os
import sys

# Protocol replies go to a private copy of stdout; fd 1 points at stderr before any import (see _wenet_worker.py).
out = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)

import json  # noqa: E402

import torch  # noqa: E402
from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # noqa: E402

config = FireRedAsr2Config(use_gpu=False, use_half=False, beam_size=3, nbest=1, decode_max_len=0, softmax_smoothing=1.25,
                           aed_length_penalty=0.6, eos_penalty=1.0, return_timestamp=False)
model = FireRedAsr2.from_pretrained("aed", sys.argv[1], config)
print(json.dumps({"ready": "cpu"}), file=out, flush=True)
for line in sys.stdin:
    try:
        with torch.inference_mode():
            r = model.transcribe(["clip"], [line.strip()])
        print(json.dumps({"text": r[0]["text"] if r else ""}, ensure_ascii=False), file=out, flush=True)
    except Exception as e:  # report and keep serving
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), file=out, flush=True)
