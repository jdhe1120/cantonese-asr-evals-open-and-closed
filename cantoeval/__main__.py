import argparse
import json
import webbrowser

from . import runner, server
from .clips import load_clips
from .models import DEFAULT, REGISTRY


NO_CLIPS = ("no clips in clips/. Audio isn't in the repo: see \"Getting the clips\" in the README. "
            "The published results are in results/, and `python -m cantoeval ui` shows them without any clips.")


def main() -> None:
    ap = argparse.ArgumentParser(prog="cantoeval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="transcribe + score")
    r.add_argument("--models", nargs="+", default=DEFAULT, choices=list(REGISTRY))
    r.add_argument("--clips", nargs="+", help="clip ids (default: all)")
    r.add_argument("--model-clips", nargs="+", default=[], metavar="MODEL=CLIP[,CLIP]",
                   help="limit a model to some clips, e.g. gemini=norway-p1")
    r.add_argument("--with-full", action="store_true",
                   help="transcribe the whole-story *-full clips too (off by default); "
                        "without it they still show cached results, outside the averages")
    r.add_argument("--force", action="store_true", help="ignore the hypothesis cache")
    r.add_argument("--only", action="store_true",
                   help="list only --models in the results (default: every other model too, from its cache, never called)")
    r.add_argument("--no-ui", action="store_true")
    r.add_argument("--port", type=int, default=8765)
    u = sub.add_parser("ui", help="serve the UI for the latest run")
    u.add_argument("--port", type=int, default=8765)
    sub.add_parser("check", help="verify clips, normalisation and deps without loading a model")
    sub.add_parser("export", help="write results/ (summary table + per-clip transcripts) from the latest run")
    a = ap.parse_args()

    if a.cmd == "check":
        from .metrics import score
        clips = load_clips()
        if not clips:
            raise SystemExit(NO_CLIPS)
        for c in clips:
            s = score(c.reference, c.reference)
            assert s["cer"] == 0 and s["refLen"] > 0, c.id
            print(f"{c.id:14s} {c.seconds:6.1f}s {s['refLen']:4d} chars")
        from .normalize import stray_radicals
        for c in clips:
            left = stray_radicals(c.reference)
            assert not left, f"{c.id}: unfolded radical characters in the reference: {left}"
        assert score("食長民青骨", "⻝⻑⺠⻘⻣")["cer"] == 0
        # Kangxi radical must score as the ordinary graph
        assert score("二零一六", "⼆零⼀六")["cer"] == 0
        print("ok: clips load, self-CER is 0, Kangxi radicals fold")
    elif a.cmd == "export":
        from .export import export
        print(f"wrote {export()}/summary.md, summary.csv, per_clip.csv")
    elif a.cmd == "ui":
        srv = server.serve(a.port)
        print(f"http://127.0.0.1:{a.port}")
        webbrowser.open(f"http://127.0.0.1:{a.port}")
        srv.serve_forever()
    else:
        if not a.no_ui:
            server.serve(a.port, background=True)
            webbrowser.open(f"http://127.0.0.1:{a.port}")
            print(f"UI: http://127.0.0.1:{a.port}")
        mc = {k: v.split(",") for k, v in (x.split("=", 1) for x in a.model_clips)}
        if not load_clips():
            raise SystemExit(NO_CLIPS)  # before anything writes runs/latest.json, which would hide the published results
        doc = runner.run(a.models, a.clips, a.force, mc, a.with_full, show_cached=not a.only)
        print(json.dumps(doc["summary"], ensure_ascii=False, indent=1))
        if not a.no_ui:
            print("run finished; UI stays up, Ctrl-C to exit")
            try:
                import time
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass


main()
