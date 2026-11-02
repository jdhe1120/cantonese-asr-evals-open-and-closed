"""Stock Qwen3-ASR-1.7B asked for traditional characters (Qwen3-ASR has no script setting: the language is forced
with the assistant prefill `language Cantonese<asr_text>`, which every Qwen row already uses, and it still writes
mostly simplified). Same weights and revision as qwen3-asr-stock. Two levers:
  prompt    text in the system slot (Qwen3-ASR's context field, documented for hotwords)
  nosimp    decoding never picks a token containing a simplified-only character (normalize.simplified_only:
            们, 这, 说, but not 后, 系, 里)
VARIANTS are the candidates, each runnable as its own experimental model qwen3-asr-trad-<key> (see models/__init__);
scripts/qwen_trad_prompts.py compares them on samples/trad100.txt. CHOSEN is the variant the qwen3-asr-trad row runs."""
import functools

from ..normalize import simplified_only
from ._qwen import transcribe as _transcribe
from .qwen_stock import MODEL_ID, REVISION

VARIANTS = {
    "instr": {"prompt": "請用香港繁體中文書寫粵語口語。"},
    "instr-en": {"prompt": "Write the transcript in Traditional Chinese characters, as written Hong Kong Cantonese."},
    "sample": {"prompt": "我哋而家喺香港。佢話聽日會嚟搵你，你係咪得閒呀？"},
    "words": {"prompt": "係 喺 嘅 咗 啲 嚟 冇 嗰 噉 佢哋 我哋 點解 乜嘢 這 個 們 說 為 會 對 時 來 還 開 關 學 聽 講 話 裏 後 發"},
    # Background-knowledge formats: Qwen3-ASR's technical report says the system prompt is read as context, not as
    # instructions, and cantocaptions-ai measured a labelled reference ("［label］：text") biasing best and an
    # imperative wrapper no better than none (its docs/asr-context.md). "Technical terms:" is a hotword prefix
    # reported to help in QwenLM/Qwen3-ASR issue #186.
    "labelled": {"prompt": "［香港粵語字幕，繁體中文］：佢哋喺度食緊飯。我哋聽日再嚟搵你，你係咪得閒呀？"},
    "passage": {"prompt": "［香港粵語字幕，繁體中文］：今日天氣好好，我哋一齊去咗海邊行下。佢話呢度啲嘢食幾好味，"
                          "不過有時要排好耐隊。你聽日得唔得閒？如果得閒嘅話，我哋可以喺地鐵站等，然後再諗下去邊度。"
                          "嗰間舖頭啱啱開咗，聽講啲人都話幾抵食。我覺得都係早啲去好，唔使等咁耐。"},
    "terms": {"prompt": "Technical terms: 係、喺、嘅、咗、啲、嚟、冇、嗰度、噉樣、佢哋、我哋、點解、乜嘢、聽日、"
                        "這個、說話、時間、還有、開始、關係、學校、對唔住、後生、發覺"},
    "nosimp": {"nosimp": True},
    "instr-en+nosimp": {"prompt": "Write the transcript in Traditional Chinese characters, as written Hong Kong Cantonese.",
                        "nosimp": True},
}
CHOSEN = "passage"  # best prompt-only variant on samples/trad100.txt (2026-10-25)


@functools.cache
def _suppress_ids() -> tuple[int, ...]:
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION).tokenizer
    bad = simplified_only()
    return tuple(i for i in range(len(tok)) if any(ch in bad for ch in tok.decode([i])))


def transcribe(wav_path: str, variant: str | None = None) -> str:
    v = VARIANTS[variant or CHOSEN]
    return _transcribe(MODEL_ID, wav_path, REVISION, v.get("prompt"), _suppress_ids() if v.get("nosimp") else None)
