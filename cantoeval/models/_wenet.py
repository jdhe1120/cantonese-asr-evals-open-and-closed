"""Shared runner for WeNet checkpoints (whisper-m-yue, u2pp-yue): one isolated worker process per model
(_wenet_worker.py), since WeNet isn't on PyPI and its pins clash with the main env. WeNet comes from the GitHub archive
of commit a54b90b (`git+` installs fail on a broken submodule entry), plus torchcodec, which its wav decoding uses.

A model dir must hold final.pt, train.yaml (with input_dim / output_dim) and units.txt: wenet.cli.model.load_model's
layout. scripts/fetch_models.py builds them in models/ (gitignored).
"""
from pathlib import Path

from ._worker import Worker

WENET = "wenet @ https://github.com/wenet-e2e/wenet/archive/a54b90bc768679bd4217e4c7765c0671fbfb3a7a.zip"
WORKER = Path(__file__).with_name("_wenet_worker.py")
MODELS = Path(__file__).resolve().parent.parent.parent / "models"


def transcriber(name: str, model_dir: Path, lang: str | None = None):
    worker = None

    def transcribe(wav_path: str) -> str:
        nonlocal worker
        if worker is None:
            missing = [f for f in ("final.pt", "train.yaml", "units.txt") if not (model_dir / f).exists()]
            if missing:
                raise RuntimeError(f"{name} not found in {model_dir} (missing {missing}): run "
                                   f"scripts/fetch_models.py {name}")
            worker = Worker(name, ["uv", "run", "--isolated", "--no-project", "--python", "3.12", "--with", WENET,
                                   "--with", "torch", "--with", "torchaudio", "--with", "torchcodec",
                                   "python", "-I", str(WORKER), str(model_dir)] + ([lang] if lang else []))
        return worker.transcribe(wav_path)

    return transcribe
