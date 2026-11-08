"""Runs models over clips, caching hypotheses and writing progress as it goes.

Every cached result is filled in first; then the models still to transcribe form one queue: local models one at a
time (each unloaded before the next loads), hosted APIs in parallel threads beside them. runs/latest.json is
rewritten as cells finish (the UI polls it, at most about once a second); on finish the
same document is also saved as runs/<runId>.json and runs/<runId>.csv.
Scores are computed here from cached text, so changing the
normalisation never needs a re-transcription.
"""
import csv
import json
import os
import threading
import time
import tomllib
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import cache, metrics
from .convert import to_hk_traditional
from .clips import Clip, clip_set_version, load_clips
from .normalize import is_simplified
from .models import COLUMNS, DERIVED, EXPERIMENTS, REGISTRY, call_gap, describe, get_pacer, get_transcriber, is_local, unload_local, uses_cache

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
LEVELS = Path(__file__).resolve().parent.parent / "rules" / "story_levels.toml"


FLAGS = Path(__file__).resolve().parent.parent / "rules" / "flags.toml"


def _flags() -> list[dict]:
    """Results to read with care (contaminated / partial), shown as markers on the page and in results/."""
    return tomllib.loads(FLAGS.read_text(encoding="utf-8")).get("flag", []) if FLAGS.exists() else []


def _levels() -> dict[str, int]:
    """Story name -> learner reading level (1-5), for the stories that have one."""
    if not LEVELS.exists():
        return {}
    return {name: s["level"] for name, s in tomllib.loads(LEVELS.read_text(encoding="utf-8"))["stories"].items()}


def _write(path: Path, doc: dict) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")  # per process: two runs at once must not swap each other's file
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# (fold interchangeable graphs, digits -> Chinese numerals, score in traditional); the UI picks one with its three
# toggles. The traditional views (t*) exist only for clips whose group has traditional references.
VARIANTS = {"strict": (False, False, False), "equiv": (True, False, False), "num": (False, True, False),
            "equivNum": (True, True, False), "tStrict": (False, False, True), "tEquiv": (True, False, True),
            "tNum": (False, True, True), "tEquivNum": (True, True, True)}


def traditional_clips(clips: list[Clip]) -> set[str]:
    """Clips that get the traditional views: their dataset is mostly traditional (the Guangzhou sets aren't), and their
    own reference has no simplified-only character (the mixed set has some simplified references; two others have
    a simplified typo)."""
    by: dict[str, list[Clip]] = {}
    for c in clips:
        by.setdefault(c.id.rpartition("-")[0], []).append(c)
    return {c.id for g in by.values() if sum(map(is_simplified, (c.reference for c in g))) <= len(g) / 2
            for c in g if not is_simplified(c.reference)}


def _scored(clip: Clip, text: str, trad: bool) -> dict:
    scores = {}
    for name, (equiv, numerals, t) in VARIANTS.items():
        if t and not trad:
            continue
        sc = metrics.score(clip.reference, text, equiv, numerals, t)
        scores[name] = {**sc, "diff": metrics.diff(clip.reference, text, equiv, numerals, t)}
    # cer / cerEquiv stay at the top level for the CSV
    return {"scores": scores, "cer": scores["strict"]["cer"], "cerEquiv": scores["equiv"]["cer"],
            "refLen": scores["strict"]["refLen"]}


def _summary(doc: dict) -> dict:
    out = {}
    clips = {c["id"]: c for c in doc["clips"]}
    for model, cells in doc["cells"].items():
        done = [c for cid, c in cells.items() if c["status"] == "done" and clips[cid]["aggregate"]]
        out[model] = {v: metrics.summarize([c["scores"][v] for c in done if v in c["scores"]]) for v in VARIANTS}
        out[model].update(clipsDone=len(done), clipsTotal=sum(c["aggregate"] for c in clips.values()))
    return out


def run(models: list[str], only_clips: list[str] | None = None, force: bool = False,
        model_clips: dict[str, list[str]] | None = None, include_full: bool = False, show_cached: bool = True) -> dict:
    """model_clips limits one model to some clips (e.g. a paid API on a single clip); its other cells are "skipped".
    The whole-story *-full clips are left out unless include_full (then listed and transcribed, never averaged)."""
    model_clips = model_clips or {}
    # show_cached: every other model is listed too, filled from the cache only (never called), so the results page keeps
    # all columns while a run works on some of them.
    shown_only = [m for m in REGISTRY if m not in models and m not in EXPERIMENTS] if show_cached else []
    models = set(models) | set(shown_only)
    derived = [d for d, base in DERIVED.items() if base in models]  # converted copies of a model's cached text
    models = sorted(models | set(derived), key=lambda m: COLUMNS.index(m) if m in COLUMNS else len(COLUMNS))
    clips = [c for c in load_clips() if (c.aggregate or include_full) and (not only_clips or c.id in only_clips)]
    trad = traditional_clips(clips)
    is_trad = lambda c: c.id in trad
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    levels = _levels()
    doc = {
        "runId": run_id, "state": "running", "startedAt": run_id, "finishedAt": None,
        "clipSetVersion": clip_set_version(clips), "models": models,
        "modelInfo": {m: describe(m) for m in models}, "current": [], "flags": _flags(),
        "normalization": "NFKC, OpenCC t2s, punctuation/space stripped; 'equiv' also folds rules/bench_conventions.toml",
        "clips": [{"id": c.id, "seconds": round(c.seconds, 1), "reference": c.reference, "aggregate": c.aggregate,
                   "level": levels.get(c.id.rpartition("-")[0]), "trad": is_trad(c)} for c in clips],
        "cells": {m: {c.id: {"status": "pending"} for c in clips} for m in models},
    }
    latest = RUNS_DIR / "latest.json"
    lock = threading.RLock()  # API models run in threads beside the local ones; every change to doc holds it
    last_flush = [0.0]

    def flush(force: bool = True) -> None:
        """Rewrite runs/latest.json, at most about once a second unless something the page shows live changed."""
        with lock:
            if not force and time.time() - last_flush[0] < 1.0:
                return
            last_flush[0] = time.time()
            doc["summary"] = _summary(doc)
            _write(latest, doc)

    flush()
    # Fill every cached result before transcribing anything, so finished numbers are on the page from the start and
    # stay put. --model-clips limits what is transcribed, not which cached results show.
    for clip in clips:
        for model in models:
            if model in DERIVED or not uses_cache(model):
                continue
            redo = force and model not in shown_only and (model not in model_clips or clip.id in model_clips[model])
            hit = None if redo else cache.get(model, clip.id, clip.digest)  # --force redoes only what it re-runs
            if hit:
                took = hit.get("seconds") or None  # batch results (scripts/gemini_batch.py) have no per-clip time
                doc["cells"][model][clip.id].update(
                    status="done", text=hit["text"], seconds=round(took, 2) if took else None, cached=True,
                    rtf=round(took / clip.seconds, 2) if took and clip.seconds else None,
                    **_scored(clip, hit["text"], is_trad(clip)))
            elif model in shown_only:  # display only and never run on this clip: final now, not "pending"
                doc["cells"][model][clip.id].update(status="skipped", note="not part of this run and no cached result")

    def derive() -> None:
        """Converted rows: the base model's text through the HK converter, scored like any cell (no transcription)."""
        with lock:
            for d in derived:
                for clip in clips:
                    base = doc["cells"][DERIVED[d]][clip.id]
                    if base["status"] == "done":
                        if doc["cells"][d][clip.id].get("source") != base["text"]:
                            text = to_hk_traditional(base["text"])
                            doc["cells"][d][clip.id] = {"status": "done", "text": text, "source": base["text"],
                                                        "cached": True, **_scored(clip, text, is_trad(clip))}
                    else:
                        doc["cells"][d][clip.id] = {"status": "skipped", "note": f"no {DERIVED[d]} result to convert"}

    derive()

    # The queue: each run model's clips still to transcribe. A paced model's --model-clips list also sets its order,
    # so one that runs out of daily quota partway has done its list's first clips.
    todo: dict[str, list[Clip]] = {}
    for model in (m for m in models if m not in shown_only and m not in DERIVED):
        order = clips
        if get_pacer(model) and model in model_clips:
            pos = {cid: i for i, cid in enumerate(model_clips[model])}
            order = sorted(clips, key=lambda c: pos.get(c.id, len(pos)))
        todo[model] = []
        for clip in order:
            cell = doc["cells"][model][clip.id]
            if cell["status"] == "done":
                continue
            if model in model_clips and clip.id not in model_clips[model]:
                cell["status"] = "skipped"
            elif not clip.aggregate and not include_full and uses_cache(model):
                cell.update(status="skipped", note="whole-story clip: run with --with-full to transcribe it")
            else:
                todo[model].append(clip)

    # Progress for the whole queue, per model: what's left and how long one transcription takes (this run's own
    # timings once there are some, else the cached ones, never less than a paced API's gap), so the page can show
    # one bar and a time estimate. Local models run one after another; APIs run alongside them.
    def pace_of(model: str, timings: list[float]) -> float | None:
        if not timings:
            timings = [c["seconds"] for c in doc["cells"][model].values() if c.get("cached") and c.get("seconds")]
        mean = sum(timings) / len(timings) if timings else None
        gap = call_gap(model)
        return max(mean or 0.0, gap) or None
    took_here: dict[str, list[float]] = {m: [] for m in todo}
    doc["work"] = {"total": sum(map(len, todo.values())), "done": 0,
                   "models": {m: {"total": len(todo[m]), "done": 0, "local": is_local(m), "state": "queued",
                                  "secPerClip": pace_of(m, [])} for m in todo if todo[m]}}
    doc["current"] = []
    flush()

    def work(model: str) -> None:
        """Transcribe one model's queue. Local models run in turn on the main thread; each API in its own thread."""
        w = doc["work"]["models"][model]
        with lock:
            w["state"] = "running"
        transcribe, pace, halted = None, get_pacer(model), None
        for clip in todo[model]:
            cell = doc["cells"][model][clip.id]
            if halted:
                with lock:
                    cell.update(status="skipped", note=halted)
                    w["done"] += 1
                    doc["work"]["done"] += 1
                continue
            with lock:
                cell["status"] = "running"
                doc["current"] = [c for c in doc["current"] if c["model"] != model] + [{"model": model, "clip": clip.id}]
            flush(force=False)
            try:
                if transcribe is None:
                    transcribe = get_transcriber(model)
                if pace:
                    pace()  # waits out the API's gap; deliberately outside the timer
                t0 = time.time()
                text = transcribe(str(clip.wav))
                took = time.time() - t0
                if uses_cache(model):
                    cache.put(model, clip.id, clip.digest, {"text": text, "seconds": took})
                with lock:
                    took_here[model].append(took)
                    cell.update(status="done", text=text, seconds=round(took, 2), cached=False,
                                rtf=round(took / clip.seconds, 2) if clip.seconds else None,
                                **_scored(clip, text, is_trad(clip)))
            except Exception as e:  # one bad cell must not end the run
                with lock:
                    if getattr(e, "skip_cell", False):
                        cell.update(status="skipped", note=str(e))
                    elif getattr(e, "halt_model", False):
                        halted = str(e)
                        cell.update(status="skipped", note=halted)
                        print(f"[{model}] stopping for this run: {e}", flush=True)
                    else:
                        cell.update(status="error", error=f"{type(e).__name__}: {e}")
                        traceback.print_exc()
            with lock:
                w["done"] += 1
                doc["work"]["done"] += 1
                w["secPerClip"] = pace_of(model, took_here[model])
            flush(force=False)
        with lock:
            w["state"] = "stopped" if halted else "done"
            doc["current"] = [c for c in doc["current"] if c["model"] != model]
        flush()

    apis = [threading.Thread(target=work, args=(m,), daemon=True) for m in todo if todo[m] and not is_local(m)]
    for t in apis:
        t.start()
    for model in (m for m in todo if todo[m] and is_local(m)):  # one local model in memory at a time
        work(model)
        unload_local()
    for t in apis:
        t.join()
    derive()
    doc.update(state="finished", current=[], finishedAt=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    flush()
    _write(RUNS_DIR / f"{run_id}.json", doc)
    with (RUNS_DIR / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "clip", "status", "cer", "cer_equiv", "ref_len", "seconds", "hypothesis"])
        for m, cells in doc["cells"].items():
            for cid, c in cells.items():
                w.writerow([m, cid, c["status"], c.get("cer"), c.get("cerEquiv"), c.get("refLen"),
                            c.get("seconds"), c.get("text", c.get("error", ""))])
    return doc
