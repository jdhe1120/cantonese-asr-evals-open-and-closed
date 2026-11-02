"""Builds clips/ from Audio/yue-0029_norway.{mp3,pdf}.

The PDF's text layer uses Kangxi radical code points (⼆ ⼗ ⽉ ⾺) that look
identical to the normal graphs but are different characters, so every model
would be charged for them. NFKC folds them. cantoeval.normalize applies NFKC
again at scoring time, so this is belt and braces.

Paragraph boundaries are the ~1.7 s pauses in the narration (ffmpeg
silencedetect). Also writes norway-full: the whole narration as one clip.
"""
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MP3 = ROOT / "Audio/yue-0029_norway.mp3"
PDF = ROOT / "Audio/yue-0029_norway.pdf"
OUT = ROOT / "clips"

# (start, end) seconds per paragraph, title excluded.
SEGS = [(3.1, 23.95), (25.6, 43.55), (45.2, 69.85), (71.45, 91.5),
        (93.2, 99.45), (101.1, 123.9), (125.6, 141.4), (143.0, 157.65)]


def paragraphs() -> list[str]:
    raw = subprocess.run(["pdftotext", str(PDF), "-"], capture_output=True, text=True, check=True).stdout
    out = []
    for page in raw.split("\f")[1:]:
        lines = [l.strip() for l in page.splitlines() if l.strip() and not l.strip().isdigit()]
        s = unicodedata.normalize("NFKC", "".join(lines))
        if "。" in s and "LIDA" not in s:
            out.append(s)
    return out


def cut(name: str, a: float, b: float, text: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a), "-to", str(b), "-i", str(MP3),
                    "-ar", "16000", "-ac", "1", str(OUT / f"{name}.wav")], check=True)
    (OUT / f"{name}.txt").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    paras = paragraphs()
    assert len(paras) == len(SEGS), (len(paras), len(SEGS))
    for i, ((a, b), t) in enumerate(zip(SEGS, paras), 1):
        cut(f"norway-p{i}", a, b, t)
    cut("norway-full", SEGS[0][0], SEGS[-1][1], "".join(paras))


if __name__ == "__main__":
    main()
