"""Step-Audio-2-mini (stepfun-ai, 8.3B, Apache-2.0), stock, on Modal (A100 40 GB), as its makers run ASR: their own
StepAudio2 class (stepaudio2.py + utils.py of stepfun-ai/Step-Audio2 at 76e272b, transformers 4.49 as their README
pins), system prompt 请记录下你所听到的语音内容。 (examples.py's asr_test), audio as the human turn, greedy.

    uv run --with modal modal run modal_apps/step_audio2.py::download    # weights into the Volume, if not there
    uv run --with modal modal run modal_apps/step_audio2.py --wav clips/mixed-p1.wav   # one-clip smoke test
"""
import modal

import os

VOLUME = os.environ.get("MODAL_WEIGHTS_VOLUME", "cantonese-asr-weights")
SCALEDOWN_S = 120  # an idle container shuts down after this

APP_NAME = "cantonese-step-audio2"
REPO, REVISION = "stepfun-ai/Step-Audio-2-mini", "e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36"
PATH = "/weights/Step-Audio-2-mini"
CODE = "https://raw.githubusercontent.com/stepfun-ai/Step-Audio2/76e272b56c3917a8d7188f18bbb5a65dfc8a0845"
SYSTEM = "请记录下你所听到的语音内容。"
GPU = "A100-40GB"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "curl")
    .pip_install("torch==2.6.0", "torchaudio==2.6.0", "transformers==4.49.0", "accelerate", "librosa", "soundfile",
                 "huggingface_hub")
    .run_commands(f"mkdir -p /opt/step && cd /opt/step && curl -sfLO {CODE}/stepaudio2.py && curl -sfLO {CODE}/utils.py")
)


@app.function(image=image, volumes={"/weights": vol}, timeout=3600, cpu=4)
def download() -> list[str]:
    import os

    from huggingface_hub import snapshot_download

    snapshot_download(REPO, revision=REVISION, local_dir=PATH, ignore_patterns=["assets/*", "token2wav/*"])
    vol.commit()
    return sorted(os.listdir(PATH))


@app.cls(image=image, volumes={"/weights": vol}, gpu=GPU, timeout=3600, cpu=4, memory=32768,
         scaledown_window=SCALEDOWN_S)
class Model:
    @modal.enter()
    def load(self):
        import sys
        import time
        sys.path.insert(0, "/opt/step")
        from stepaudio2 import StepAudio2

        t0 = time.time()
        self.m = StepAudio2(PATH)
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
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "human", "content": [{"type": "audio", "audio": f.name}]},
                    {"role": "assistant", "content": None}]
            _, text, _ = self.m(msgs, max_new_tokens=512, do_sample=False)
            out[cid] = (text.strip(), time.time() - t0)
            os.unlink(f.name)
        return out


@app.local_entrypoint()
def main(wav: str):
    from pathlib import Path

    p = Path(wav)
    print(Model().transcribe.remote([(p.stem, p.read_bytes())]))
