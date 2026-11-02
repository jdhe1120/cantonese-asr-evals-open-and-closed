"""Builds clips/ from Audio/yue-NNNN_<name>.{mp3,pdf} (LIDA Stories narrations).

Each story is a cover page, N text pages (one narrated "page" each), and a credits
page. The narrator pauses ~1.7 s between pages, so the N longest internal pauses
are the title|page1 ... page(N-1)|pageN boundaries. A story is only written when
those N pauses are clearly longer than every other pause; otherwise it is reported
and skipped rather than cut in the wrong place.

References come from the PDF text layer, NFKC-normalised (it uses Kangxi radical
code points such as ⼆ ⼗ ⽉ that look identical to the real graphs).

Clips: <name>-p1..pN (one per page, title excluded) and <name>-full (all pages as
one clip). A story whose <name>-p1.wav already exists is left alone, so clips that
were cut earlier (norway) keep their cache.

  python scripts/prepare_stories.py --check     # print pauses and margins only
  python scripts/prepare_stories.py             # write clips
"""
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "Audio"
OUT = ROOT / "clips"
PAD = 0.15
MIN_MARGIN = 1.25  # shortest boundary pause / longest non-boundary pause


def pages(pdf: Path) -> list[str]:
    raw = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    out = []
    for page in raw.split("\f")[1:]:
        lines = [l.strip() for l in page.splitlines() if l.strip() and not l.strip().isdigit()]
        s = unicodedata.normalize("NFKC", "".join(lines))
        if "。" in s and "LIDA" not in s:
            out.append(s)
    return out


def duration(mp3: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(mp3)], capture_output=True, text=True, check=True).stdout)


def silences(mp3: Path, total: float) -> list[tuple[float, float]]:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(mp3), "-af", "silencedetect=noise=-35dB:d=0.4",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    ends += [total] * (len(starts) - len(ends))  # silence that runs to the end of the file
    return list(zip(starts, ends))


def plan(name: str, mp3: Path, pdf: Path):
    """-> (segments, texts, margin) or raises ValueError explaining why not."""
    texts = pages(pdf)
    total = duration(mp3)
    inner = [(a, b) for a, b in silences(mp3, total) if a > 0.3 and b < total - 0.05]
    if len(inner) < len(texts):
        raise ValueError(f"{len(inner)} internal pauses for {len(texts)} pages")
    ranked = sorted(inner, key=lambda s: s[1] - s[0], reverse=True)
    boundaries = sorted(ranked[:len(texts)])
    rest = ranked[len(texts):]
    shortest = min(b - a for a, b in boundaries)
    longest_other = max((b - a for a, b in rest), default=0.0)
    margin = shortest / longest_other if longest_other else float("inf")
    # speech runs: [title][p1]...[pN]; the title is the run before the first boundary
    edges = [(a, b) for a, b in boundaries]
    starts = [b - PAD for _, b in edges]
    ends = [a + PAD for a, _ in edges[1:]] + [total]
    last_speech_end = [a for a, b in silences(mp3, total) if b >= total - 0.05]
    ends[-1] = (last_speech_end[-1] if last_speech_end else total) + PAD
    segs = list(zip(starts, ends))
    return segs, texts, margin, [round(b - a, 2) for a, b in ranked[:len(texts) + 3]]


def cut(mp3: Path, a: float, b: float, name: str, text: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(mp3),
                    "-ar", "16000", "-ac", "1", str(OUT / f"{name}.wav")], check=True)
    (OUT / f"{name}.txt").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    check = "--check" in sys.argv
    OUT.mkdir(exist_ok=True)
    for mp3 in sorted(AUDIO.glob("yue-*.mp3")):
        name = mp3.stem.split("_", 1)[1]
        pdf = mp3.with_suffix(".pdf")
        if (OUT / f"{name}-p1.wav").exists():
            print(f"{name:10s} already cut, skipping")
            continue
        try:
            segs, texts, margin, top = plan(name, mp3, pdf)
        except ValueError as e:
            print(f"{name:10s} SKIPPED: {e}")
            continue
        ok = margin >= MIN_MARGIN
        print(f"{name:10s} {len(texts):2d} pages  margin x{margin:.1f}  "
              f"{'ok' if ok else 'UNCLEAR, not written'}  pauses {top}")
        if check or not ok:
            continue
        for i, ((a, b), t) in enumerate(zip(segs, texts), 1):
            cut(mp3, a, b, f"{name}-p{i}", t)
        cut(mp3, segs[0][0], segs[-1][1], f"{name}-full", "".join(texts))


if __name__ == "__main__":
    main()
