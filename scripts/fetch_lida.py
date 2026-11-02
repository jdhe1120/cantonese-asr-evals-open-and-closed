"""Fills clips/ with the LIDA story pages from the published dataset, as <story>-p<N>.wav + .txt.

  uv run python scripts/fetch_lida.py

No account needed. prepare_stories.py and prepare_norway.py are how these clips were cut from the original mp3s and PDFs.
"""
import csv
from pathlib import Path

import soundfile as sf
from huggingface_hub import snapshot_download

REPO = "jdhe1120/cantonese-lida-stories-speech"
OUT = Path(__file__).resolve().parent.parent / "clips"


def main() -> None:
    root = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns="pages/test/*")) / "pages" / "test"
    OUT.mkdir(exist_ok=True)
    rows = list(csv.DictReader((root / "metadata.csv").open(encoding="utf-8")))
    for r in rows:
        audio, rate = sf.read(root / r["file_name"], dtype="int16")
        clip = OUT / f"{r['story']}-p{r['page']}"
        sf.write(clip.with_suffix(".wav"), audio, rate, subtype="PCM_16")
        clip.with_suffix(".txt").write_text(r["text"] + "\n", encoding="utf-8")
    print(f"{len(rows)} clips in {OUT}")


if __name__ == "__main__":
    main()
