"""Writes a small, committable snapshot of the latest run to results/: the summary table and every clip's transcripts.

runs/ holds the full documents (with diffs, hundreds of MB) and stays out of git; results/ is what the repo shows.
  results/summary.md   mean CER per group for each model in all eight views, simplified first, ranked by average
  results/ranking.svg  the README chart: the headline view, ranked
  results/summary.csv  the same numbers, one row per group x model x view
  results/per_clip.csv every model's transcript and CER on every scored clip, with the edit counts behind each of the
                       four views, so the UI can rebuild a run from it (cantoeval/snapshot.py) when runs/ is empty
"""
import csv
import json
from pathlib import Path

from . import metrics
from .models import COLUMNS, EXPERIMENTS
from .runner import RUNS_DIR

OUT = Path(__file__).resolve().parent.parent / "results"
NAMED = {"commonvoice": "Common Voice", "narration": "Narration", "cantomap": "CantoMap (conversation, noisy references)",
         "mixed": "Mixed Cantonese–English", "wordshk": "words.hk sentences", "gzdaily": "Guangzhou daily use",
         "gzcabin": "Guangzhou in-car"}
GROUPS = [(label, k, lambda g, k=k: g == k) for k, label in NAMED.items()] + \
    [("LIDA stories", "stories", lambda g: g not in NAMED)]  # last, so its level columns open at the end
MARK = {"contaminated": "⚠️", "partial": "◐"}
README = OUT.parent / "README.md"
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"  # the README table between these is regenerated
# The first view of each list is the headline for that script; the README shows those two.
TRAD_VIEWS = [("tEquivNum", "traditional, graphs folded, digits → 中文"), ("tEquiv", "traditional, graphs folded"),
              ("tNum", "traditional, digits → 中文"), ("tStrict", "traditional, strict")]
VIEWS = [("equivNum", "simplified, graphs folded, digits → 中文"), ("equiv", "simplified, graphs folded"),
         ("num", "simplified, digits → 中文"), ("strict", "simplified, strict (the community benchmark's scoring)")]
# per_clip.csv column suffix for each scoring view (the trad_* ones are empty where the reference is simplified)
COLS = {"strict": "strict", "equiv": "folded", "num": "num", "equivNum": "folded_num",
        "tStrict": "trad_strict", "tEquiv": "trad_folded", "tNum": "trad_num", "tEquivNum": "trad_folded_num"}


def _public(doc: dict, hide_models: list[str]) -> dict:
    """The run without the hidden models (their columns, cells, descriptions and flags)."""
    keep = sorted((m for m in doc["models"] if m not in hide_models and m not in EXPERIMENTS),  # current column order, not the run's
                  key=lambda m: COLUMNS.index(m) if m in COLUMNS else len(COLUMNS))
    return {**doc, "models": keep, "cells": {m: doc["cells"][m] for m in keep},
            "modelInfo": {m: doc["modelInfo"][m] for m in keep},
            "flags": [f for f in doc.get("flags", []) if f["model"] not in hide_models]}


def _rows(doc: dict) -> list[tuple[str, str, list[dict]]]:
    """(label, group id, clips); story levels follow the stories row with an indented label and the stories' id."""
    clips = [c for c in doc["clips"] if c["aggregate"]]
    group_of = lambda c: c["id"].rpartition("-")[0]
    out = []
    for label, gid, match in GROUPS:
        members = [c for c in clips if match(group_of(c))]
        if not members:
            continue
        out.append((label, gid, members))
        if gid == "stories":
            for level in sorted({c["level"] for c in members if c.get("level")}):
                out.append((f"  level {level}", gid, [c for c in members if c.get("level") == level]))
    return out


# Short column headers; the full names are in the README's test data section.
# Task names; the README's Test data table maps them to the community benchmark's.
SHORT = {"Common Voice": "Many Speakers", "Narration": "Storytelling", "CantoMap (conversation, noisy references)": "Conversation",
         "Mixed Cantonese–English": "Canto + Eng Mixed", "words.hk sentences": "Synthetic Voice", "Guangzhou daily use": "GZ Daily Use",
         "Guangzhou in-car": "GZ Car Commands", "LIDA stories": "Stories by Level"}
# Shown but never ranked: its model card lists two test sets as training data, and it has no stock baseline beside it.
# Unlike a ranked model's flags, its flags don't remove a group from everyone's average.
UNRANKED = {"whisper-yue"}


def _table(cols, models, label, cell, mark, view, is_open, average=None, left_out=()) -> list[str]:
    """Models as rows, groups as columns headed with their clip counts. With `average` (model -> mean or None), rows
    are ranked by it and models it can't rank go last; without it, open models come first, then closed APIs.
    Groups in left_out (ids) are marked † in the header: the Average skips them."""
    head = lambda g, members: (f"{SHORT.get(g.strip(), g.strip()).replace('level ', 'LIDA L')} ({len(members)})"
                               + (" †" if any(gid in left_out for x, gid, _ in cols if x == g) else ""))
    first = "| Model | Type | Average | " if average else "| Model | Type | "
    lines = [first + " | ".join(head(g, mem) for g, _, mem in cols) + " |",
             "|---|---|" + ("---|" if average else "") + "---|" * len(cols)]
    if average:
        models = sorted(models, key=lambda m: (average(m) is None, average(m) or 0))
    else:
        models = sorted(models, key=lambda m: not is_open(m))
    for m in models:
        kind = "open" if is_open(m) else "closed"
        mean = "" if not average else ("unranked" if average(m) is None else f"**{100 * average(m):.1f}%**") + " | "
        lines.append(f"| {label(m)} | {kind} | {mean}" + " | ".join(
            (c := cell(g.strip(), m, view)) + (mark(m, gid) if c != "–" else "") for g, gid, _ in cols) + " |")
    return lines


def _chart(rows: list[tuple[str, float, bool]], subtitle: str, note: str) -> str:
    """Ranked horizontal bars as a standalone SVG on its own light surface, readable on GitHub's light and dark pages.
    rows: (label, percent, is open), best first."""
    # 8 px a character allows for wider fallback fonts: GitHub viewers don't all have the same one
    left, bar_max, top, step, thick = max(len(r[0]) for r in rows) * 8 + 32, 330, 104, 30, 18
    width, height = left + bar_max + 72, top + step * len(rows) + (34 if note else 12)
    scale = bar_max / max(v for _, v, _ in rows)
    ink, muted = "#0b0b0b", "#52514e"
    colour = {True: "#2a78d6", False: "#eb6834"}
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;")
    text = lambda x, y, s, size=13, fill=ink, anchor="start", weight=400: (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'role="img" aria-label="{esc(subtitle)}" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
           f'<rect width="{width}" height="{height}" rx="8" fill="#fcfcfb"/>',
           text(20, 32, "Character error rate: lower is better", 17, weight=600),
           text(20, 54, subtitle, 13, muted)]
    for x, is_open, name in ((20, True, "Open model"), (120, False, "Closed API")):
        out += [f'<rect x="{x}" y="66" width="12" height="12" rx="3" fill="{colour[is_open]}"/>',
                text(x + 18, 77, name, 13, muted)]
    for i, (name, value, is_open) in enumerate(rows):
        y, w = top + i * step, max(value * scale, 6)
        out += [text(left - 12, y + 14, name, 13, anchor="end"),
                f'<path d="M{left} {y}h{w - 4:.1f}a4 4 0 0 1 4 4v{thick - 8}a4 4 0 0 1 -4 4h-{w - 4:.1f}z" fill="{colour[is_open]}"/>',
                text(left + w + 8, y + 14, f"{value:.1f}%", 13)]
    if note:
        out.append(text(20, height - 14, note, 12, muted))
    return "\n".join(out + ["</svg>"]) + "\n"


def export(src: Path = RUNS_DIR / "latest.json", out: Path = OUT, readme: Path = README,
           hide_models: list[str] = (), no_text_groups: list[str] = ()) -> Path:
    """hide_models are left out entirely; no_text_groups keep their scores but not the reference or transcript text
    (datasets whose license doesn't allow republishing it)."""
    doc = _public(json.loads(src.read_text(encoding="utf-8")), list(hide_models))
    models, cells, info = doc["models"], doc["cells"], doc["modelInfo"]
    out.mkdir(exist_ok=True)
    rows = _rows(doc)
    label = lambda m: info[m]["label"].split(" (")[0]
    is_open = lambda m: bool(info[m].get("hf"))  # open weights on Hugging Face; the rest are closed APIs
    groups = [r for r in rows if not r[0].startswith("  ")]
    levels = [r for r in rows if r[0].startswith("  ")]
    trad_only = lambda m: bool(info[m].get("tradOnly"))  # converted rows: traditional tables only
    simp_models = [m for m in models if not trad_only(m)]
    trad_models = [m for m in models if not info[m].get("simpOnly")]  # mostly-simplified writers: converted row only
    # traditional tables: each group's clips with a traditional reference; groups left with none drop out
    trad_only_clips = lambda rs: [(g, gid, t) for g, gid, mem in rs if (t := [c for c in mem if c.get("trad")])]
    trad_rows = trad_only_clips(groups)

    summary = []  # (group, model, view, mean, corpus, n, N)
    blocks = [(views, group, members) for views, rs in ((VIEWS, rows), (TRAD_VIEWS, trad_only_clips(rows)))
              for group, _, members in rs]
    for views, group, members in blocks:
        for m in models:
            done = [cells[m][c["id"]] for c in members if cells[m][c["id"]]["status"] == "done"]
            for view, _ in views:
                s = metrics.summarize([d["scores"][view] for d in done if view in d["scores"]])
                summary.append((group.strip(), m, view, s["meanCer"], s["corpusCer"], s["n"], len(members)))

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "model", "view", "mean_cer", "corpus_cer", "clips_done", "clips_in_group"])
        for g, m, v, mean, corpus, n, total in summary:
            w.writerow([g, m, v, None if mean is None else round(mean, 4), None if corpus is None else round(corpus, 4),
                        n, total])

    lookup = {(g, m, v): (mean, n, total) for g, m, v, mean, _, n, total in summary}
    cell = lambda g, m, v: (lambda mean, n, total: "–" if mean is None else
                            f"{100 * mean:.1f}%" + ("" if n == total else f" ({n}/{total})"))(*lookup[(g, m, v)])
    flags = doc.get("flags", [])
    flag_of = lambda m, g: next((f for f in flags if f["model"] == m and f["group"] in (g, "*")), None)
    mark = lambda m, g: f" {MARK.get(f['kind'], '*')}" if (f := flag_of(m, g)) else ""
    legend = ["", "⚠️ very likely trained on this test data: don't compare it on that group. "
              "◐ covers only part of the group: its average isn't comparable. "
              "† left out of every Average, so all models are ranked on the same data."]
    gname = lambda g: "all groups" if g == "*" else "LIDA stories" if g == "stories" else NAMED.get(g, g)
    legend += [f"- {MARK.get(f['kind'], '*')} **{label(f['model'])}**, {gname(f['group'])}: "
               f"{f['note']}" for f in flags]

    # any ranked model: dropped for every model
    trained_on = {f["group"] for f in flags if f["kind"] == "contaminated" and f["model"] not in UNRANKED}

    def average(view, cols):
        """Rough guide: the plain mean over the groups no ranked model seems trained on, the same groups for each."""
        clean = [c for c in cols if c[1] not in trained_on]

        def of(m):
            means = [lookup[(g.strip(), m, view)][0] for g, _, _ in clean]
            return None if m in UNRANKED or any(x is None for x in means) else sum(means) / len(means)
        return of

    trad_view, simp_view = TRAD_VIEWS[0][0], VIEWS[0][0]
    headline = [m for m in trad_models if f"{m}+hk" not in models]  # a raw row only where no converted row replaces it
    table = lambda rs, ms, view, avg=True: _table(rs, ms, label, cell, mark, view, is_open,
                                                  average(view, rs) if avg else None, trained_on if avg else ())
    md = [f"# Results\n\nRun `{doc['runId']}`, clip set `{doc['clipSetVersion']}`. Average clip CER (the mean of "
          "per-clip CER, as in the community benchmark); groups are scored separately, with each column's clip count "
          "in its header. Average is the plain mean of the groups no ranked model seems trained on, a rough guide. A gap of a "
          "point or two is within noise. Generated by `python -m cantoeval export`.\n"]
    md += legend[1:] + [""]
    md.append("## Simplified\n")
    md.append("Everything converted to simplified first, as in the community benchmark, so a model is never penalised "
              "for its script.\n")
    for i, (view, view_label) in enumerate(VIEWS):
        md += [f"### {view_label.capitalize()}\n"] + table(groups, simp_models, view) + [""]
        if i == 0 and levels:
            md += ["LIDA stories by learner level:\n"] + table(levels, simp_models, view, False) + [""]
    md += ["## Traditional\n", "Scored without converting to simplified, so a simplified character is "
           "an error; only groups whose references are traditional. Hong Kong and Taiwan variant spellings count the "
           "same. Rows ending in \"+ HK converter\" are that model's transcripts converted to Hong Kong traditional "
           "(cantoeval/convert.py); they replace Qwen3-ASR (stock) and SenseVoice, which often write simplified "
           "(unconverted they score 30-40% here; summary.csv has those numbers).\n"]
    for i, (view, view_label) in enumerate(TRAD_VIEWS):
        md += [f"### {view_label.capitalize()}\n"] + table(trad_rows, trad_models, view) + [""]
        if i == 0 and levels:
            md += ["LIDA stories by learner level:\n"] + table(trad_only_clips(levels), trad_models, view, False) + [""]
    md.append("## Models\n")
    md += [f"- **{label(m)}**: {info[m].get('detail', '')}" for m in models]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    avg = average(simp_view, groups)
    ranked = sorted(((label(m), 100 * avg(m), is_open(m)) for m in simp_models if avg(m) is not None),
                    key=lambda r: r[1])
    clean = [g for g, gid, _ in groups if gid not in trained_on]
    dropped = [SHORT.get(g, g) for g, gid, _ in groups if gid in trained_on]
    (out / "ranking.svg").write_text(_chart(
        ranked, f"Scored in simplified, graphs folded, digits as Chinese numerals. Average of {len(clean)} groups.",
        f"Left out: {', '.join(dropped)} (a ranked model appears trained on them). Unranked: "
        + ", ".join(label(m) for m in simp_models if m in UNRANKED) + "."),
        encoding="utf-8")

    def best_by_task() -> list[str]:
        """Per group, the best ranked model not flagged there, and any others within a point of it."""
        lines = ["| Task | Best | Within a point |", "|---|---|---|"]
        for g, gid, _ in groups:
            scored = sorted((mean, m) for m in simp_models if m not in UNRANKED and not flag_of(m, gid)
                            and (mean := lookup[(g.strip(), m, simp_view)][0]) is not None)
            top = scored[0][0]
            near = [f"{label(m)} {100 * x:.1f}%" for x, m in scored[1:3] if x - top <= 0.01]
            lines.append(f"| {SHORT.get(g, g)} | **{label(scored[0][1])}** {100 * top:.1f}% | {', '.join(near) or '–'} |")
        return lines

    # README: the task table, headline chart and tables, between the markers
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if BEGIN in text and END in text:
            block = [BEGIN, *best_by_task(), "",
                     "Scored in simplified. The winners are the same models in traditional, where Qwen needs a converter; "
                     "the GZ tasks have no traditional references. A model flagged on a task isn't listed for it.", "",
                     "## Overall", "", "![Models ranked by character error rate](results/ranking.svg)", "",
                     "### Simplified", "", *table(groups, simp_models, simp_view), "",
                     "### Traditional", "", *table(trad_rows, headline, trad_view), "",
                     "Percentages are character error rates, graphs folded, digits → 中文. ⚠️ very likely trained on "
                     "this test data. † left out of every Average. All eight scoring combinations and every "
                     f"transcript are in [`results/`](results/summary.md) (run `{doc['runId']}`).", END]
            head, rest = text.split(BEGIN, 1)
            readme.write_text(head + "\n".join(block) + rest.split(END, 1)[1], encoding="utf-8")

    with (out / "per_clip.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "seconds", "reference", "model", "cer_folded", "cer_strict", "transcript"]
                   + [f"{k}_{v}" for v in COLS.values() for k in ("edits", "chars")])
        for c in doc["clips"]:
            if not c["aggregate"]:
                continue
            text_ok = c["id"].rpartition("-")[0] not in no_text_groups
            for m in models:
                x = cells[m][c["id"]]
                if x["status"] == "done":
                    sc = x["scores"]
                    w.writerow([c["id"], c["seconds"], c["reference"] if text_ok else "", m,
                                round(sc["equiv"]["cer"], 4), round(sc["strict"]["cer"], 4), x["text"] if text_ok else ""]
                               + [sc[k][f] if k in sc else "" for k in COLS for f in ("edits", "refLen")])
    return out
