"""Model registry. Each module exposes `transcribe(wav_path) -> str`.
Imported lazily so listing models never loads torch."""
import importlib
from typing import Callable

# name -> module under cantoeval.models
REGISTRY = {
    "cantocaptions": "cantocaptions",  # v2, Sep 20 weights (pinned)
    "cantocaptions-v3": "cantocaptions_v3",  # Oct 18 weights (pinned)
    "qwen3-asr-stock": "qwen_stock",
    "qwen3-asr-trad": "qwen_trad",  # stock weights, prompted for traditional characters (qwen_trad.CHOSEN)
    "sensevoice": "sensevoice",
    "whisper-yue": "whisper_yue",
    "glm-asr": "glm_asr",
    "glm-asr-yue": "glm_asr_yue",
    "azure-standard": "azure_standard",
    "mai": "mai",
    "gemini": "gemini",
    "cantonese-ai": "cantonese_ai",
}
# Experiments: run only when named (never listed from cache alongside a run, never exported). "module:arg" passes arg
# to the module's transcribe; the cache folder is the name, as for any model.
EXPERIMENTS = {f"qwen3-asr-trad-{v}": f"qwen_trad:{v}" for v in
               ("instr", "instr-en", "sample", "words", "labelled", "passage", "terms", "nosimp", "instr-en+nosimp")}
REGISTRY.update(EXPERIMENTS)
# Converted rows: a model's cached text through the HK traditional converter (cantoeval/convert.py), computed by the
# runner, never transcribed. Shown only in the traditional view, where they stand in for what an app would display.
DERIVED = {"qwen3-asr-stock+hk": "qwen3-asr-stock", "qwen3-asr-trad+hk": "qwen3-asr-trad", "sensevoice+hk": "sensevoice"}
DEFAULT = ["cantocaptions", "sensevoice"]
# Column order in results: open models left, closed (paid / hosted) right.
COLUMNS = ["cantocaptions-v3", "cantocaptions", "qwen3-asr-stock", "qwen3-asr-stock+hk", "qwen3-asr-trad",
           "qwen3-asr-trad+hk", "sensevoice", "sensevoice+hk", "whisper-yue", "glm-asr", "glm-asr-yue",
           "azure-standard", "gemini", "mai"]
COLUMNS += ["cantonese-ai"]


def _module(name: str):
    return importlib.import_module(f"cantoeval.models.{REGISTRY[name].partition(':')[0]}")


def get_transcriber(name: str) -> Callable[[str], str]:
    mod, _, arg = REGISTRY[name].partition(":")
    f = importlib.import_module(f"cantoeval.models.{mod}").transcribe
    return (lambda wav: f(wav, arg)) if arg else f


def uses_cache(name: str) -> bool:
    """False for models that just read a local file (re-reading it is free and picks up edits)."""
    return getattr(_module(name), "USE_CACHE", True)


def get_pacer(name: str) -> Callable[[], None] | None:
    """Optional `pace()` hook: blocks until the model's API will accept another call."""
    return getattr(_module(name), "pace", None)


def is_local(name: str) -> bool:
    """Runs on this machine (open weights on Hugging Face); the rest are hosted APIs."""
    return bool(INFO[name].get("hf"))


def call_gap(name: str) -> float:
    """Seconds a paced API waits between calls (optional `gap_s()` hook), for the time estimate."""
    f = getattr(_module(name), "gap_s", None)
    return f() if f else 0.0


def unload_local() -> None:
    """Drop every loaded local model, so the next one gets the memory (the runner does local models one at a time)."""
    import gc
    import sys
    for mod, attr in (("_qwen", "_loaded"), ("sensevoice", "_model"), ("glm_asr", "_loaded"), ("whisper_yue", "_pipe")):
        m = sys.modules.get(f"cantoeval.models.{mod}")
        if m is not None:
            setattr(m, attr, {} if isinstance(getattr(m, attr), dict) else None)
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Display names. `label` is the short name in tables, `detail` says exactly what ran.
# `hf` is the Hugging Face repo, so the run can record the cached revision it used.
INFO = {
    "cantocaptions": {
        "label": "CantoCaptions v2 (Sep 20; Qwen3-ASR-1.7B, LoRA-tuned)",
        "detail": "rookes/cantocaptions-cantonese-asr at d9e2680 (Sep 20 2026 weights): Qwen/Qwen3-ASR-1.7B-hf + LoRA "
                  "rank 128, one epoch on ~75 h of CantoCaptions audio, merged weights.",
        "hf": "rookes/cantocaptions-cantonese-asr",
        "revision": "d9e2680",
    },
    "cantocaptions-v3": {
        "label": "CantoCaptions v3 (Oct 18; Qwen3-ASR-1.7B, LoRA-tuned)",
        "detail": "rookes/cantocaptions-cantonese-asr at aa5f1f3 (Oct 18 2026 weights, commit \"lora-v6: raw-audio share, "
                  "written-out repeats, vocal-noise clips\"). Same base model and loading as v2.",
        "hf": "rookes/cantocaptions-cantonese-asr",
        "revision": "aa5f1f3",
    },
    "qwen3-asr-stock": {
        "label": "Qwen3-ASR-1.7B (stock)",
        "detail": "Qwen/Qwen3-ASR-1.7B-hf at bcd2b5b, untuned baseline.",
        "hf": "Qwen/Qwen3-ASR-1.7B-hf",
        "revision": "bcd2b5b",
    },
    "sensevoice": {
        "label": "SenseVoice-Small",
        "detail": "FunAudioLLM/SenseVoiceSmall at 3847d57 (July 2024 release, no later version number) + fsmn-vad at df20e6b, "
                  "run with FunASR, language yue, ITN on.",
        "hf": "FunAudioLLM/SenseVoiceSmall",
        "revision": "3847d57",
    },
    "gemini": {
        "label": "Gemini 3.5 Transcribe",
        "detail": "gemini-3.5-transcribe via the Gemini API (free and paid tiers), verbatim mode, language yue-Hant-HK.",
        "hf": None,
    },
    "whisper-yue": {
        "label": "JackyHoCL Whisper fine-tune",
        "detail": "JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english at 58d693f, language yue, "
                  "repetition penalty 1.3.",
        "hf": "JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english",
        "revision": "58d693f",
    },
    "glm-asr": {
        "label": "GLM-ASR-Nano-2512",
        "detail": "zai-org/GLM-ASR-Nano-2512 at 61ba4e0 (1.5B, Dec 2025), stock, via transformers' GlmAsrForConditionalGeneration, "
                  "default prompt (no language setting), greedy decoding.",
        "hf": "zai-org/GLM-ASR-Nano-2512",
        "revision": "61ba4e0",
    },
    "glm-asr-yue": {
        "label": "GLM-ASR-Nano-2512 + Cantonese prompt",
        "detail": "zai-org/GLM-ASR-Nano-2512 at 61ba4e0, stock weights, prompt 請將呢段粵語錄音逐字轉寫成粵語口語。 in place of the default "
                  "instruction, greedy decoding.",
        "hf": "zai-org/GLM-ASR-Nano-2512",
        "revision": "61ba4e0",
    },
    "azure-standard": {
        "label": "Azure Speech standard (zh-HK)",
        "detail": "Azure Speech fast transcription with no enhancedMode (the regular locale-specific recognizer, not MAI), "
                  "locale zh-HK, api-version 2025-10-15. Microsoft doesn't name the model; newest zh-HK base model on "
                  "the resource is 20260811.",
        "hf": None,
    },
    "mai": {
        "label": "MAI-Transcribe-2",
        "detail": "Microsoft MAI-Transcribe-2 (public preview) via Azure Speech fast transcription, "
                  "api-version 2025-10-15, language yue, verbatim.",
        "hf": None,
    },
    "cantonese-ai": {
        "label": "cantonese.ai",
        "detail": "cantonese.ai (closed source) via its synchronous /api/stt endpoint, default settings (fusion off, "
                  "no timestamps or diarization). Lite plan (5 requests a minute), all clips.",
        "hf": None,
    },
}


INFO["qwen3-asr-trad"] = {**INFO["qwen3-asr-stock"], "label": "Qwen3-ASR-1.7B prompted for traditional",
                          "detail": "Qwen/Qwen3-ASR-1.7B-hf at bcd2b5b (same weights as the stock row) with a system "
                                    "prompt asking for traditional characters; the prompt is qwen_trad.CHOSEN in "
                                    "cantoeval/models/qwen_trad.py, picked on a 100-clip test."}
for _name, _base in DERIVED.items():
    INFO[_name] = {**INFO[_base], "label": INFO[_base]["label"].split(" (")[0] + " + HK converter", "tradOnly": True,
                   "detail": f"{_base}'s transcripts converted to Hong Kong traditional (OpenCC s2t plus "
                             "cantocaptions-ai's Cantonese variant table; cantoeval/convert.py). Traditional view only."}
# Models that mostly write simplified: in the traditional view their "+ HK converter" row stands in for them
for _name in ("qwen3-asr-stock", "sensevoice"):
    INFO[_name]["simpOnly"] = True
for _name, _spec in EXPERIMENTS.items():
    _v = _spec.partition(":")[2]
    INFO[_name] = {**INFO["qwen3-asr-stock"], "simpOnly": False, "label": f"Qwen3-ASR-1.7B trad test: {_v}",
                   "detail": f"Experiment: stock Qwen3-ASR-1.7B at bcd2b5b asked for traditional characters, variant "
                             f"'{_v}' (cantoeval/models/qwen_trad.py)."}


def describe(name: str) -> dict:
    """INFO entry plus the Hugging Face revision found in the local cache, if any."""
    from pathlib import Path

    info = dict(INFO[name])
    if info.get("hf") and not info.get("revision"):  # pinned models state their revision in INFO
        ref = Path.home() / ".cache/huggingface/hub" / ("models--" + info["hf"].replace("/", "--")) / "refs/main"
        info["revision"] = ref.read_text().strip()[:7] if ref.exists() else None
    return info
