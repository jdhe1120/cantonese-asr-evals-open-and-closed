"""Transcribe a Modal model's uncached clips in one ephemeral Modal app, into cache/<model>/.

  uv run --with modal python scripts/modal_batch.py MODEL [--limit N] [--clips a,b] [--chunk 150]

MODEL is a registry name whose module (cantoeval/models/<module>.py) names its Modal app (APP, a file in modal_apps/)
and a parse(text) -> str. One container is started and the model loaded once; clips go in chunks to that warm
container and each chunk is cached as it comes back, so an interruption loses at most one chunk. Only clips with no
cached result are sent, never whole-story -full clips. The app stops when the script ends (nothing stays deployed).

The cache entry's `seconds` is the GPU time for that clip (model loading and container start not included; for batched
models, the batch's time spread over its clips). Afterwards: uv run python -m cantoeval run --no-ui --models MODEL
(all cached, so no calls) to rebuild the run.
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modal_apps"))
from cantoeval import cache  # noqa: E402
from cantoeval.clips import load_clips  # noqa: E402
from cantoeval.models import REGISTRY  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--limit", type=int)
    p.add_argument("--clips")
    p.add_argument("--chunk", type=int, default=150)
    args = p.parse_args()
    mod = importlib.import_module(f"cantoeval.models.{REGISTRY[args.model]}")
    remote = importlib.import_module(mod.APP)
    only = set(args.clips.split(",")) if args.clips else None
    todo = [c for c in load_clips() if c.aggregate and not cache.exists(args.model, c.id, c.digest)
            and (only is None or c.id in only)][:args.limit]
    if not todo:
        sys.exit("nothing to send: every clip has a cached result")
    print(f"[{args.model}] {len(todo)} clips, {sum(c.seconds for c in todo) / 3600:.2f} h of audio, "
          f"chunks of {args.chunk}", flush=True)
    t_start, done = time.time(), 0
    with remote.app.run():
        model = remote.Model()
        for i in range(0, len(todo), args.chunk):
            chunk = todo[i:i + args.chunk]
            out = model.transcribe.remote([(c.id, Path(c.wav).read_bytes()) for c in chunk])
            for c in chunk:
                raw, took = out[c.id]
                cache.put(args.model, c.id, c.digest, {"text": mod.parse(raw), "raw": raw, "seconds": round(took, 2),
                                                        "modal": remote.APP_NAME})
            done += len(chunk)
            print(f"[{args.model}] {done}/{len(todo)} cached, {(time.time() - t_start) / 60:.1f} min", flush=True)
    print(f"[{args.model}] done in {(time.time() - t_start) / 60:.1f} min wall time", flush=True)


if __name__ == "__main__":
    main()
