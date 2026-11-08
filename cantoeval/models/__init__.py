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
    "sensevoice-yue": "sensevoice_yue",  # WenetSpeech-Yue models (ASLP-lab/WSYue-ASR; scripts/fetch_models.py)
    "whisper-m-yue": "whisper_m_yue",
    "u2pp-yue": "u2pp_yue",
    "fireredasr2": "fireredasr2",
    "mimo-asr": "mimo_asr",  # Modal GPU, filled by scripts/modal_batch.py
    "step-audio2": "step_audio2",
    "qwen3-omni": "qwen3_omni",
    "glm-asr": "glm_asr",
    "glm-asr-yue": "glm_asr_yue",
    "azure-standard": "azure_standard",
    "mai": "mai",
    "gemini": "gemini",
    "gemini-3.8-flash": "gemini_flash",  # general models prompted to transcribe (same prompt, gemini_argon.PROMPT)
    "gemini-argon": "gemini_argon",
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
           "qwen3-asr-trad+hk", "sensevoice", "sensevoice+hk", "sensevoice-yue", "whisper-yue", "whisper-m-yue", "u2pp-yue",
           "fireredasr2", "mimo-asr", "step-audio2", "qwen3-omni", "glm-asr", "glm-asr-yue",
           "azure-standard", "gemini", "gemini-3.8-flash", "gemini-argon", "mai"]
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
    from ._worker import stop_all
    stop_all()  # isolated worker processes (WeNet models, FireRedASR2)
    for mod, attr in (("_qwen", "_loaded"), ("sensevoice", "_model"), ("sensevoice_yue", "_model"), ("glm_asr", "_loaded"), ("whisper_yue", "_pipe")):
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
    "sensevoice-yue": {
        "label": "SenseVoice-Small-Yue (WenetSpeech-Yue)",
        "detail": "ASLP-lab/WSYue-ASR at 4a74a91, sensevoice_small_yue: SenseVoiceSmall fine-tuned on WenetSpeech-Yue "
                  "(ASLP@NPU, 2025). Run like the stock row: FunASR + fsmn-vad at df20e6b, language yue, ITN on.",
        "hf": "ASLP-lab/WSYue-ASR",
        "revision": "4a74a91",
    },
    "whisper-m-yue": {
        "label": "Whisper-medium-Yue (WenetSpeech-Yue)",
        "detail": "ASLP-lab/WSYue-ASR at 4a74a91, whisper_medium_yue: Whisper medium (769M) fine-tuned on WenetSpeech-Yue, "
                  "WeNet a54b90b, CPU, attention decoding.",
        "hf": "ASLP-lab/WSYue-ASR",
        "revision": "4a74a91",
    },
    "u2pp-yue": {
        "label": "Conformer-U2pp-Yue (WenetSpeech-Yue)",
        "detail": "ASLP-lab/WSYue-ASR at 4a74a91, u2pp_conformer_yue: 130M Conformer U2++ trained on WenetSpeech-Yue, "
                  "WeNet a54b90b, CPU, default decode, fbank dither off.",
        "hf": "ASLP-lab/WSYue-ASR",
        "revision": "4a74a91",
    },
    "fireredasr2": {
        "label": "FireRedASR2-AED",
        "detail": "FireRedTeam/FireRedASR2-AED at 2304afe (Mar 2026), stock, fireredasr2s code at 4e7d9aa, CPU, the README's "
                  "AED decoding (beam 3). Lowercases English.",
        "hf": "FireRedTeam/FireRedASR2-AED",
        "revision": "2304afe",
    },
    "mimo-asr": {
        "label": "MiMo-V2.5-ASR",
        "detail": "XiaomiMiMo/MiMo-V2.5-ASR at 227ce4c (Apr 2026, 8B) + MiMo-Audio-Tokenizer at 8a41b55, stock, the repo's "
                  "MimoAudio.asr_sft (code 876312a) with automatic language detection, on a Modal A100.",
        "hf": "XiaomiMiMo/MiMo-V2.5-ASR",
        "revision": "227ce4c",
    },
    "step-audio2": {
        "label": "Step-Audio-2-mini",
        "detail": "stepfun-ai/Step-Audio-2-mini at e36fdd5 (8.3B), stock, the makers' StepAudio2 code (76e272b) and ASR "
                  "system prompt 请记录下你所听到的语音内容。, greedy, on a Modal A100. Leading language tag dropped.",
        "hf": "stepfun-ai/Step-Audio-2-mini",
        "revision": "e36fdd5",
    },
    "qwen3-omni": {
        "label": "Qwen3-Omni-30B-A3B (prompted)",
        "detail": "Qwen/Qwen3-Omni-30B-A3B-Instruct at 26291f7, stock, thinker only, prompt 请将这段粤语语音转换为纯文本。 "
                  "(the card's ASR prompt naming Cantonese), greedy, on a Modal A100 80 GB.",
        "hf": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "revision": "26291f7",
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
    "gemini-argon": {
        "label": "Gemini 4 Argon (prompted)",
        "detail": "gemini-4-argon via generateContent, audio inline with a prompt asking for verbatim spoken Cantonese "
                  "in traditional characters (gemini_argon.PROMPT), default sampling.",
        "hf": None,
    },
    "gemini-3.8-flash": {
        "label": "Gemini 3.8 Flash (prompted)",
        "detail": "gemini-3.8-flash via generateContent with the same prompt and request as gemini-argon "
                  "(gemini_argon.PROMPT), default sampling.",
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
for _name, _spec in ((n, s) for n, s in EXPERIMENTS.items() if s.startswith("qwen_trad:")):
    _v = _spec.partition(":")[2]
    INFO[_name] = {**INFO["qwen3-asr-stock"], "simpOnly": False, "label": f"Qwen3-ASR-1.7B trad test: {_v}",
                   "detail": f"Experiment: stock Qwen3-ASR-1.7B at bcd2b5b asked for traditional characters, variant "
                             f"'{_v}' (cantoeval/models/qwen_trad.py)."}
# Parameter counts of the open models (model cards; FireRedASR2's from its load log). Closed APIs don't publish theirs.
PARAMS = {"cantocaptions": "1.7B", "cantocaptions-v3": "1.7B", "qwen3-asr-stock": "1.7B", "sensevoice": "234M",
          "sensevoice-yue": "234M", "whisper-yue": "809M", "whisper-m-yue": "769M", "u2pp-yue": "130M",
          "fireredasr2": "1.15B", "glm-asr": "1.5B", "glm-asr-yue": "1.5B", "mimo-asr": "8B", "step-audio2": "8.3B",
          "qwen3-omni": "30B (3B active)"}
PARAMS["qwen3-asr-trad"] = PARAMS["qwen3-asr-stock"]
PARAMS.update({_name: PARAMS[_base] for _name, _base in DERIVED.items() if _base in PARAMS})
for _name, _p in PARAMS.items():
    INFO[_name]["params"] = _p


def describe(name: str) -> dict:
    """INFO entry plus the Hugging Face revision found in the local cache, if any."""
    from pathlib import Path

    info = dict(INFO[name])
    if info.get("hf") and not info.get("revision"):  # pinned models state their revision in INFO
        ref = Path.home() / ".cache/huggingface/hub" / ("models--" + info["hf"].replace("/", "--")) / "refs/main"
        info["revision"] = ref.read_text().strip()[:7] if ref.exists() else None
    return info
