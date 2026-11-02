"""Builds the LIDA stories clips as a Hugging Face audio dataset, locally, for review before any upload.

Layout (Hugging Face "audiofolder": the viewer and load_dataset work without a loading script):
  hf/cantonese-lida-stories-speech/
    README.md                         dataset card (license, credit, changes, levels, limits)
    pages/test/<story>-pNN.flac       one clip per story page        + pages/test/metadata.csv
    full_stories/test/<story>.flac    each whole story as one clip   + full_stories/test/metadata.csv

Audio is the 16 kHz mono clips in clips/, stored as FLAC (lossless). Text is the page text from the story PDFs,
NFKC-normalised, with the PDFs' CJK Radicals Supplement look-alikes (⻝ ⻑ ...) mapped to the ordinary characters;
punctuation is kept. Levels and titles come from rules/story_levels.toml.

This script never uploads. After checking the folder, upload with:
  huggingface-cli upload jdhe1120/cantonese-lida-stories-speech hf/cantonese-lida-stories-speech --repo-type dataset
"""
import csv
import shutil
import sys
import tomllib
import unicodedata
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cantoeval.normalize import _RADICALS  # noqa: E402

REPO = "jdhe1120/cantonese-lida-stories-speech"
OUT = ROOT / "hf" / REPO.split("/")[1]
CLIPS = ROOT / "clips"
FIELDS = ["file_name", "text", "story", "page", "level", "title_zh", "title_en", "duration"]


def clean(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_RADICALS).strip()


def write(src: Path, dest: Path) -> float:
    audio, sr = sf.read(str(src))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), audio, sr, format="FLAC")
    return round(len(audio) / sr, 2)


def main() -> None:
    stories = tomllib.loads((ROOT / "rules" / "story_levels.toml").read_text(encoding="utf-8"))["stories"]
    if OUT.exists():
        shutil.rmtree(OUT)
    pages, fulls = [], []
    for name, s in sorted(stories.items(), key=lambda kv: (kv[1]["level"], kv[0])):
        meta = {"story": name, "level": s["level"], "title_zh": s["title_zh"], "title_en": s["title_en"]}
        nums = sorted(int(p.stem.rsplit("-p", 1)[1]) for p in CLIPS.glob(f"{name}-p*.wav"))
        if not nums:
            print(f"warning: no clips for {name}")
            continue
        for n in nums:
            src = CLIPS / f"{name}-p{n}.wav"
            rel = f"{name}-p{n:02d}.flac"
            dur = write(src, OUT / "pages" / "test" / rel)
            pages.append({"file_name": rel, "text": clean(src.with_suffix(".txt").read_text(encoding="utf-8")),
                          "page": n, "duration": dur, **meta})
        full = CLIPS / f"{name}-full.wav"
        if full.exists():
            rel = f"{name}.flac"
            dur = write(full, OUT / "full_stories" / "test" / rel)
            fulls.append({"file_name": rel, "text": clean(full.with_suffix(".txt").read_text(encoding="utf-8")),
                          "page": None, "duration": dur, **meta})
    for subset, rows in (("pages", pages), ("full_stories", fulls)):
        with (OUT / subset / "test" / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    hours = sum(r["duration"] for r in pages) / 3600
    by_level = {lv: sum(r["level"] == lv for r in pages) for lv in range(1, 6)}
    (OUT / "README.md").write_text(card(len(pages), len(fulls), len({r["story"] for r in pages}), hours, by_level,
                                        stories), encoding="utf-8")
    print(f"wrote {OUT}: {len(pages)} page clips ({hours * 60:.1f} min), {len(fulls)} full stories; "
          f"pages per level {by_level}")


def card(n_pages: int, n_full: int, n_stories: int, hours: float, by_level: dict, stories: dict) -> str:
    rows = "\n".join(f"| {s['level']} | {s['title_en']} | {s['title_zh']} | `{k}` |"
                     for k, s in sorted(stories.items(), key=lambda kv: (kv[1]["level"], kv[0])))
    levels = ", ".join(f"level {lv}: {n}" for lv, n in by_level.items())
    return f"""---
license: cc-by-4.0
language:
- yue
task_categories:
- automatic-speech-recognition
pretty_name: Cantonese LIDA Stories Speech
size_categories:
- n<1K
tags:
- cantonese
- speech
- language-learning
configs:
- config_name: pages
  default: true
  data_files:
  - split: test
    path: pages/test/**
- config_name: full_stories
  data_files:
  - split: test
    path: full_stories/test/**
---

# Cantonese LIDA Stories Speech

Graded Cantonese (Hong Kong) storybooks for adult learners, read aloud, cut into one clip per page with the page
text as the transcript.

## Purpose: evaluation by learner level, in one fixed voice

This dataset is for measuring how well a speech-recognition model handles Cantonese learner material **at each reading
level (1 to 5), with the voice held constant**. Every clip is read by the same narrator, so differences between levels
come from the material (vocabulary, sentence length and complexity), not from who is speaking.

It is **not** a test of how a model copes with many different speakers, accents, ages or recording conditions. For
that, use a multi-speaker set such as Common Voice, or conversational speech such as CantoMap. A model's score here
says "how it does on graded learner Cantonese read clearly by one speaker", nothing more.

- **{n_pages} page clips** ({hours * 60:.0f} minutes) from **{n_stories} stories**, reading levels 1 to 5 ({levels})
- **{n_full} whole-story clips** in a separate subset (`full_stories`), the same audio as the pages joined up
- One narrator throughout; clean, unhurried read speech; 16 kHz mono FLAC
- Written Cantonese (粵文) transcripts in traditional characters, punctuation kept

```python
from datasets import load_dataset
pages = load_dataset("{REPO}", "pages", split="test")
```

## Source and license

The story texts and narrations are from **[LIDA Stories Hong Kong](https://lidastories.net)
(LIDA故事香港)**, narrated by **Stephanie Hung**, © LIDA故事香港, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This dataset is shared under the same license. LIDA
(Learning Inclusion in a Digital Age) makes free illustrated stories for immigrant and refugee language learners.

**Changes made:** each story's narration was cut into one clip per page at the narrator's pauses between pages
(about 1.7 s), the title page was left out, and audio was resampled to 16 kHz mono. Page text was extracted from the
story PDFs and normalised: Unicode NFKC, plus mapping the PDFs' look-alike radical characters (e.g. ⻝ → 食) to the
ordinary ones. Stories whose page pauses were ambiguous were not included.

## Columns

| Column | Meaning |
|---|---|
| `audio` | the clip |
| `text` | transcript (the page text) |
| `story` | story id |
| `page` | page number within the story (empty for whole stories) |
| `level` | reading level, 1 (easiest) to 5 |
| `title_zh`, `title_en` | story title |
| `duration` | seconds |

## Stories

| Level | Title | 標題 | id |
|---|---|---|---|
{rows}

## Limitations

- **One voice, by design.** See Purpose: results don't generalise to other speakers, accents or recording conditions.
- **Level isn't acoustic difficulty.** Low-level pages are very short (a few words), so one wrong character is a large
  share of a clip's error rate; in our tests models did better on levels 4-5 than on 2-3.
- **Small.** {n_pages} clips is enough to compare models broadly, not to separate models a point or two apart.
- **Test split only.** Meant for evaluation; there is no training split.

Prepared with cantonese-asr-evals-open-and-closed, a small Cantonese speech-recognition eval harness.
"""


if __name__ == "__main__":
    main()
