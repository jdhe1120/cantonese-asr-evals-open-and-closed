"""MiMo-V2.5-ASR (Xiaomi, Apr 2026, 8B, MIT) on Modal (A100 40 GB), as its README runs it: the repo's MimoAudio class
(XiaomiMiMo/MiMo-V2.5-ASR code at 876312a, torch 2.6 / transformers 4.49 / flash-attn 2.7.4.post1 as pinned there),
asr_sft() with no language tag (automatic detection, which the README recommends for code-switching). Text decoding
is greedy (the repo's "asr" sampler config). Needs MiMo-Audio-Tokenizer too.

    uv run --with modal modal run modal_apps/mimo_asr.py::download    # both repos into the Volume (~36 GB)
    uv run --with modal modal run modal_apps/mimo_asr.py --wav clips/mixed-p1.wav
"""
import modal

import os

VOLUME = os.environ.get("MODAL_WEIGHTS_VOLUME", "cantonese-asr-weights")
SCALEDOWN_S = 120  # an idle container shuts down after this

APP_NAME = "cantonese-mimo-asr"
REPO, REVISION = "XiaomiMiMo/MiMo-V2.5-ASR", "227ce4c5e64e356007ad948781f0e8b55a1beba0"
TOK, TOK_REVISION = "XiaomiMiMo/MiMo-Audio-Tokenizer", "8a41b556798a8cd9d83c79dcd5f48ac8e319ff0b"
PATH, TOK_PATH = "/weights/MiMo-V2.5-ASR", "/weights/MiMo-Audio-Tokenizer"
CODE = "https://github.com/XiaomiMiMo/MiMo-V2.5-ASR/archive/876312a76ce0b06c381d8b962decbc082526a0c0.tar.gz"
FLASH = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
         "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl")
GPU = "A100-40GB"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "curl")
    .pip_install("torch==2.6.0", "torchaudio==2.6.0", "transformers==4.49.0", "accelerate>=1.9.0", "librosa>=0.11.0",
                 "scipy", "triton==3.2.0", "zhon==2.1.1", "soundfile", "huggingface_hub")
    .pip_install(FLASH)
    .run_commands(f"mkdir -p /opt/mimo && curl -sfL {CODE} | tar xz -C /opt/mimo --strip-components=1")
)


@app.function(image=image, volumes={"/weights": vol}, timeout=7200, cpu=4)
def download() -> list[str]:
    import os

    from huggingface_hub import snapshot_download

    snapshot_download(REPO, revision=REVISION, local_dir=PATH)
    snapshot_download(TOK, revision=TOK_REVISION, local_dir=TOK_PATH)
    vol.commit()
    return sorted(os.listdir(PATH)) + sorted(os.listdir(TOK_PATH))


@app.cls(image=image, volumes={"/weights": vol}, gpu=GPU, timeout=3600, cpu=4, memory=32768,
         scaledown_window=SCALEDOWN_S)
class Model:
    @modal.enter()
    def load(self):
        import os
        import sys
        import time

        os.chdir("/opt/mimo")
        sys.path.insert(0, "/opt/mimo")
        from src.mimo_audio.mimo_audio import MimoAudio

        t0 = time.time()
        self.m = MimoAudio(model_path=PATH, mimo_audio_tokenizer_path=TOK_PATH)
        print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    @modal.method()
    def transcribe(self, items: list[tuple[str, bytes]]) -> dict[str, tuple[str, float]]:
        """(clip id, wav bytes) -> {clip id: (text, GPU seconds)}."""
        import os
        import tempfile
        import time

        out = {}
        for cid, data in items:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data)
            t0 = time.time()
            out[cid] = (self.m.asr_sft(f.name).strip(), time.time() - t0)
            os.unlink(f.name)
        return out


@app.local_entrypoint()
def main(wav: str):
    from pathlib import Path

    p = Path(wav)
    print(Model().transcribe.remote([(p.stem, p.read_bytes())]))
