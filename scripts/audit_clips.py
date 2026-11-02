"""Ranks a group's clips by how likely their reference or audio is wrong, from cached results only (no model runs).

A clip is suspect when the models agree it doesn't match its reference, not when one model struggles:
  best    the best model's CER on the clip. If even the best of several models is far off, suspect the clip.
  edges   every model misses the reference's first (start) or last (end) characters: words cut off by the clip edges.
  rate    reference units per second of audio (a CJK character or a whole Latin word each), far outside normal speech
          (CHARS_PER_S): text and audio mismatch.
  other   (CantoMap only) the best model's output fits a different line of the same recording better than its own:
          the clip's timestamps point at the wrong audio.
Recordings with several suspect clips are listed at the end.

Verdicts from listening go in rules/<group>_review.toml (keyed by source file name); reviewed clips are listed last
with their verdict. Use it to pick what to listen to, not to drop clips automatically: dropping clips because models fail on them makes
the test set easier in a way that depends on the models.

  python scripts/audit_clips.py cantomap               # flagged and reviewed clips
  python scripts/audit_clips.py cantomap --all --csv runs/audit_cantomap.csv
"""
import argparse
import csv
import json
import re
import statistics
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cantoeval import cache  # noqa: E402
from cantoeval.clips import load_clips  # noqa: E402
from cantoeval.metrics import diff, score  # noqa: E402
from cantoeval.normalize import normalize  # noqa: E402
from cantoeval.models import REGISTRY  # noqa: E402

BEST_CER = 0.6           # the best model still this far off
CHARS_PER_S = (1.0, 9.0)  # plausible spoken Cantonese, reference characters per second of audio
MIN_MODELS = 3


def cantomap_lines(source: str) -> dict[str, str]:
    """Every line of the recording a CantoMap clip came from (safecantonese/cantomap raw transcript), by file name."""
    from huggingface_hub import hf_hub_download

    name = source.split(":", 1)[1]
    sess = name.rsplit("_ts", 1)[0]
    path = hf_hub_download("safecantonese/cantomap", f"transcript/yue/raw/{sess}.tsv", repo_type="dataset")
    with open(path, encoding="utf-8", newline="") as f:
        return {r["path"]: r["text"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("group", help="clip id prefix, e.g. cantomap, commonvoice, narration")
    ap.add_argument("--all", action="store_true", help="print every clip, not just the top 40")
    ap.add_argument("--csv", help="also write the full table here")
    a = ap.parse_args()

    sources_path = ROOT / "clips" / "public_sources.json"
    sources = {c["id"]: c for c in json.loads(sources_path.read_text(encoding="utf-8")).get(a.group, [])} \
        if sources_path.exists() else {}
    review_path = ROOT / "rules" / f"{a.group}_review.toml"  # verdicts from listening, by source file name
    review = tomllib.loads(review_path.read_text(encoding="utf-8"))["segments"] if review_path.exists() else {}
    rows = []
    for clip in load_clips():
        if not clip.id.startswith(a.group + "-") or not clip.aggregate:
            continue
        hyps = {m: h["text"] for m in REGISTRY if (h := cache.get(m, clip.id, clip.digest))}
        if len(hyps) < MIN_MODELS:
            continue
        cers = {m: score(clip.reference, t, equiv=True)["cer"] for m, t in hyps.items()}
        diffs = [diff(clip.reference, t, equiv=True) for t in hyps.values()]
        best_model = min(cers, key=cers.get)
        # spoken units per second: each CJK character is one, and so is each run of Latin letters or digits (an
        # English word like "temperature" is one word spoken, not eleven)
        units = len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", normalize(clip.reference, equiv=True)))
        rate = units / clip.seconds if clip.seconds else 0
        flags = []
        if cers[best_model] > BEST_CER:
            flags.append("best")
        if all(d and d[0][0] == "del" for d in diffs):
            flags.append("start")
        if all(d and d[-1][0] == "del" for d in diffs):
            flags.append("end")
        if not CHARS_PER_S[0] <= rate <= CHARS_PER_S[1]:
            flags.append("rate")
        src = sources.get(clip.id, {}).get("source", "")
        other = ""
        if a.group == "cantomap" and src and cers[best_model] > 0.4:
            own = src.split(":", 1)[1]
            lines = cantomap_lines(src)
            hyp = hyps[best_model]
            alt = min((p for p in lines if p != own and len(lines[p]) > 3),
                      key=lambda p: score(lines[p], hyp)["cer"], default=None)
            if alt and score(lines[alt], hyp)["cer"] < min(0.35, cers[best_model] - 0.3):
                flags.append("other")
                other = alt
        rows.append({"clip": clip.id, "best": round(100 * cers[best_model]), "median": round(100 * statistics.median(cers.values())),
                     "models": len(hyps), "chars_per_s": round(rate, 1), "flags": " ".join(flags), "matches_line": other,
                     "recording": src.split(":", 1)[1].rsplit("_ts", 1)[0] if "_ts" in src else "",
                     "reviewed": review.get(src.split(":", 1)[1], "") if src else "",
                     "reference": clip.reference})
    # unreviewed suspects first (more flags, then a worse best model); clips someone already listened to go last
    rows.sort(key=lambda r: (bool(r["reviewed"]), -len(r["flags"].split()), -r["best"]))
    shown = rows if a.all else [r for r in rows if r["flags"] or r["reviewed"]]
    print(f"{a.group}: {len(rows)} clips with >= {MIN_MODELS} models' results; "
          f"{sum(bool(r['flags']) for r in rows)} flagged")
    print(f"{'clip':16} {'best':>5} {'median':>6} {'ch/s':>5}  flags  [verdict from listening]")
    for r in shown:
        if not r["flags"] and not r["reviewed"] and not a.all:
            continue  # unflagged and unreviewed: nothing to look at
        extra = f"  (audio matches {r['matches_line']})" if r["matches_line"] else ""
        extra += f"  [{r['reviewed']}]" if r["reviewed"] else ""
        print(f"{r['clip']:16} {r['best']:>4}% {r['median']:>5}% {r['chars_per_s']:>5}  {r['flags']}{extra}")
    by_rec = defaultdict(list)
    for r in rows:
        if r["recording"]:
            by_rec[r["recording"]].append(r)
    bad_recs = {k: v for k, v in by_rec.items() if sum("best" in r["flags"] for r in v) >= 2}
    if bad_recs:
        print("\nrecordings with 2+ clips the best model gets badly wrong:")
        for k, v in sorted(bad_recs.items(), key=lambda kv: -sum("best" in r["flags"] for r in kv[1])):
            n_bad = sum("best" in r["flags"] for r in v)
            print(f"  {k}: {n_bad}/{len(v)} sampled clips")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
