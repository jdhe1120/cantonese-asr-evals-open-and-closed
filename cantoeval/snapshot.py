"""Rebuilds a run document for the UI from the committed results/per_clip.csv, for a fresh clone with no runs/.

The CSV keeps each cell's transcript and edit counts; the diffs (most of runs/latest.json's size) are recomputed here.
Clips whose text isn't published (no reference in the CSV) keep their scores and get no diff.
"""
import csv
from collections import defaultdict
from pathlib import Path

from . import metrics
from .export import COLS
from .models import COLUMNS, describe
from .runner import VARIANTS, _flags, _levels, _summary

CSV = Path(__file__).resolve().parent.parent / "results" / "per_clip.csv"


def from_results(path: Path = CSV) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None
    levels = _levels()
    clips, cells = {}, defaultdict(dict)
    for r in rows:
        cid, ref, text = r["clip"], r["reference"], r["transcript"]
        trad = bool(r.get("edits_trad_strict"))  # empty where the reference is simplified, or in an older CSV
        clips.setdefault(cid, {"id": cid, "seconds": float(r["seconds"]), "reference": ref, "aggregate": True,
                               "level": levels.get(cid.rpartition("-")[0]), "trad": trad})
        scores = {}
        for view, (equiv, numerals, t) in VARIANTS.items():
            if t and not trad:
                continue
            edits, chars = int(r[f"edits_{COLS[view]}"]), int(r[f"chars_{COLS[view]}"])
            scores[view] = {"edits": edits, "refLen": chars, "cer": edits / chars if chars else 0.0,
                            "diff": metrics.diff(ref, text, equiv, numerals, t) if ref else None}
        cells[r["model"]][cid] = {"status": "done", "text": text, "cached": True, "scores": scores}
    models = sorted(cells, key=lambda m: COLUMNS.index(m) if m in COLUMNS else len(COLUMNS))
    doc = {"runId": "results/per_clip.csv", "state": "finished", "current": None, "clipSetVersion": "",
           "models": models, "modelInfo": {m: describe(m) for m in models}, "flags": _flags(),
           "clips": list(clips.values()), "snapshot": True,
           "cells": {m: {cid: cells[m].get(cid, {"status": "skipped", "note": "no result in results/per_clip.csv"})
                         for cid in clips} for m in models}}
    doc["summary"] = _summary(doc)
    return doc
