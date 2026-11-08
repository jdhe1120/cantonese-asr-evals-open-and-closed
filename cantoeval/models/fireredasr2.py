"""FireRedASR2-AED (FireRedTeam, Mar 2026; Apache-2.0): attention encoder-decoder for Mandarin, English and 20+ Chinese
dialects, Cantonese (Hong Kong and Guangdong) among them. Stock, in an isolated worker (_firered_worker.py) because its
package pins clash with the main env. Writes lowercase English (the code lowercases every hypothesis).

Code: fireredasr2s from the GitHub archive of 4e7d9aa (Jun 2 2026). Weights: FireRedTeam/FireRedASR2-AED at 2304afe,
downloaded on first use. The README says inputs up to 60 s; every scored clip is shorter.
"""
from pathlib import Path

from ._worker import Worker

REPO, REVISION = "FireRedTeam/FireRedASR2-AED", "2304afed56eacfee6256dee5937ed22ffa0b64ec"
CODE = "fireredasr2s @ https://github.com/FireRedTeam/FireRedASR2S/archive/4e7d9aaf4482a47cec1724807026b9b151926eb5.zip"
WORKER = Path(__file__).with_name("_firered_worker.py")
_worker = None


def transcribe(wav_path: str) -> str:
    global _worker
    if _worker is None:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(REPO, revision=REVISION)
        _worker = Worker("fireredasr2", ["uv", "run", "--isolated", "--no-project", "--python", "3.12", "--with", CODE,
                                         "--with", "kaldi_native_fbank", "python", "-I", str(WORKER), model_dir])
    return _worker.transcribe(wav_path)
