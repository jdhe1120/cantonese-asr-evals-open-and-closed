"""Qwen3-Omni-30B-A3B-Instruct (Qwen, Sep 2025) on Modal (A100 80 GB), as a transcriber: audio in, text out, the
speech-output half (talker) never loaded. The card lists Cantonese among its 19 speech-input languages but has no
language setting, so the language is in the prompt: the card's Chinese ASR prompt (请将这段中文语音转换为纯文本。)
with 中文 changed to 粤语.

Runs as on the card: user turn = audio, then the prompt text, no system prompt; greedy; transformers. 30B MoE
(3B active), bf16 thinker ~62 GB. Clips go BATCH at a time.

    uv run --with modal modal run modal_apps/qwen3_omni.py::download    # weights into the Volume, if not there (~70 GB)
    uv run --with modal modal run modal_apps/qwen3_omni.py --wav clips/mixed-p1.wav
"""
import modal

import os

VOLUME = os.environ.get("MODAL_WEIGHTS_VOLUME", "cantonese-asr-weights")
SCALEDOWN_S = 120  # an idle container shuts down after this

APP_NAME = "cantonese-qwen3-omni"
REPO, REVISION = "Qwen/Qwen3-Omni-30B-A3B-Instruct", "26291f793822fb6be9555850f06dfe95f2d7e695"  # 2025-09-22
PATH = "/weights/Qwen3-Omni-30B-A3B-Instruct"
PROMPT = "请将这段粤语语音转换为纯文本。"
GPU = "A100-80GB"
BATCH = 8

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "pillow", "transformers==5.19.0", "accelerate", "librosa",
                 "soundfile", "huggingface_hub")  # torchvision/pillow: the processor loads its image half anyway
)


@app.function(image=image, volumes={"/weights": vol}, timeout=7200, cpu=4)
def download() -> list[str]:
    import os

    from huggingface_hub import snapshot_download

    snapshot_download(REPO, revision=REVISION, local_dir=PATH)
    vol.commit()
    return sorted(os.listdir(PATH))


@app.cls(image=image, volumes={"/weights": vol}, gpu=GPU, timeout=3600, cpu=4, memory=65536,
         scaledown_window=SCALEDOWN_S)
class Model:
    @modal.enter()
    def load(self):
        import time

        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        t0 = time.time()
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            PATH, enable_audio_output=False, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        self.model.eval()
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(PATH)
        msg = [{"role": "user", "content": [{"type": "audio", "audio": "clip.wav"}, {"type": "text", "text": PROMPT}]}]
        self.text = self.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    @modal.method()
    def transcribe(self, items: list[tuple[str, bytes]]) -> dict[str, tuple[str, float]]:
        """(clip id, wav bytes) -> {clip id: (text, GPU seconds spread over its batch)}."""
        import io
        import time

        import librosa
        import torch

        out = {}
        for i in range(0, len(items), BATCH):
            chunk, t0 = items[i:i + BATCH], time.time()
            audio = [librosa.load(io.BytesIO(b), sr=16000)[0] for _, b in chunk]
            inputs = self.proc(text=[self.text] * len(chunk), audio=audio, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device)
            inputs = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
            with torch.inference_mode():
                gen = self.model.generate(**inputs, max_new_tokens=512, do_sample=False, return_audio=False,
                                          thinker_return_dict_in_generate=False)
            gen = gen[0] if isinstance(gen, tuple) else gen
            texts = self.proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            per = (time.time() - t0) / len(chunk)
            for (cid, _), t in zip(chunk, texts):
                out[cid] = (t.strip(), per)
        return out


@app.local_entrypoint()
def main(wav: str):
    from pathlib import Path

    p = Path(wav)
    print(Model().transcribe.remote([(p.stem, p.read_bytes())]))
