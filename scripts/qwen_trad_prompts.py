"""Which variant gets stock Qwen3-ASR to write traditional characters? Runs every candidate in
cantoeval/models/qwen_trad.py VARIANTS on samples/trad100.txt and compares them with plain stock Qwen, as written
and converted (cantoeval/convert.py). Run them through `cantoeval run` instead to watch progress on the page; the
cache is shared, so this script then only reads it.

  uv run python scripts/qwen_trad_prompts.py [VARIANT ...]

Each variant's outputs are cached under cache/qwen3-asr-trad-<variant>/ like any model's, so re-running is free and
the winner's 100 clips carry over to the full qwen3-asr-trad run (copy its folder to cache/qwen3-asr-trad/).
Reported per variant: share of output characters that are simplified-only, CER scored the usual way (both sides to
simplified) and CER in traditional (no conversion; HK and Taiwan variant spellings folded), and how many outputs
contain a run of the prompt itself (the model echoing its context instead of transcribing).
"""
import sys
import time
import unicodedata
from pathlib import Path

import opencc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cantoeval import cache  # noqa: E402
from cantoeval.clips import load_clips  # noqa: E402
from cantoeval.metrics import _levenshtein  # noqa: E402
from cantoeval.convert import to_hk_traditional  # noqa: E402
from cantoeval.models import qwen_trad  # noqa: E402
from cantoeval.normalize import _RADICALS  # noqa: E402

SAMPLE = ROOT / "samples" / "trad100.txt"
t2s, t2hk = opencc.OpenCC("t2s"), opencc.OpenCC("t2hk")


def chars(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).translate(_RADICALS)
                   if unicodedata.category(ch)[0] not in "PSZC")


def simplified_share(text: str) -> float:
    """Share of CJK characters that t2s leaves alone but that differ from their traditional form (s2t changes)."""
    cjk = [ch for ch in chars(text) if "一" <= ch <= "鿿"]
    s2t = opencc.OpenCC("s2t")
    return sum(ch != s2t.convert(ch) for ch in cjk) / len(cjk) if cjk else 0.0


def cer(pairs, fold) -> float:
    e = n = 0
    for ref, hyp in pairs:
        r, h = fold(chars(ref)), fold(chars(hyp))
        e += _levenshtein(r, h)
        n += len(r)
    return e / n


def echoes(text: str, prompt: str) -> bool:
    p = chars(prompt)
    return any(p[i:i + 6] in chars(text) for i in range(0, max(1, len(p) - 5)))


def main() -> None:
    ids = [l.strip() for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    clips = {c.id: c for c in load_clips()}
    clips = [clips[i] for i in ids]
    variants = sys.argv[1:] or list(qwen_trad.VARIANTS)
    stock = [cache.get("qwen3-asr-stock", c.id, c.digest)["text"] for c in clips]
    results = {"stock": stock, "stock, converted": [to_hk_traditional(s) for s in stock]}
    for v in variants:
        name, prompt, out = f"qwen3-asr-trad-{v}", qwen_trad.VARIANTS[v].get("prompt"), []
        t0, ran = time.time(), 0
        for c in clips:
            hit = cache.get(name, c.id, c.digest)
            if not hit:
                s = time.time()
                text = qwen_trad.transcribe(str(c.wav), v)
                hit = {"text": text, "seconds": time.time() - s}
                cache.put(name, c.id, c.digest, hit)
                ran += 1
            out.append(hit["text"])
        print(f"{v}: {ran} transcribed, {len(clips) - ran} cached, {time.time() - t0:.0f}s", flush=True)
        results[v] = out

    refs = [c.reference for c in clips]
    print(f"\n{len(clips)} clips from {SAMPLE.name}")
    print(f"{'variant':22s} {'simplified chars':>16s} {'CER simplified':>15s} {'CER traditional':>16s} {'echoes prompt':>14s}")
    for v, out in results.items():
        share = sum(map(simplified_share, out)) / len(out)
        prompt = qwen_trad.VARIANTS.get(v, {}).get("prompt", "")
        echo = sum(echoes(o, prompt) for o in out) if prompt else 0
        print(f"{v:22s} {share * 100:15.1f}% {cer(zip(refs, out), t2s.convert) * 100:14.1f}% "
              f"{cer(zip(refs, out), t2hk.convert) * 100:15.1f}% {echo:14d}")


if __name__ == "__main__":
    main()
