"""Shared plumbing for Gemini models called through generateContent (gemini-argon, gemini-3.8-flash).

Each model module defines MODEL, NAME (state-file prefix), build_request(wav_path) -> dict and parse(text) -> str, then
binds pace/transcribe with `make(module)`. Keys and key choice are gemini.py's (GEMINI_USE=free for the free key);
state files are cache/<NAME>_*.<key>, since Google's quotas are per model. scripts/gemini_batch.py sends the same
requests through the Batch API.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from cantoeval.models import gemini as g

API = "https://generativelanguage.googleapis.com"
QuotaExhausted = g.QuotaExhausted
gap_s = g.gap_s


def inline_wav(wav_path: str) -> dict:
    import base64
    return {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(Path(wav_path).read_bytes()).decode()}}


def response_text(resp: dict, name: str) -> str:
    """Joined non-thought text of the first candidate; raises if there is none."""
    cands = resp.get("candidates") or []
    if not cands:
        raise RuntimeError(f"{name}: no candidates ({json.dumps(resp.get('promptFeedback'))[:200]})")
    return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []) if not p.get("thought"))


def make(mod):
    """pace() and transcribe() for a model module (see the module docstring)."""
    cap = int(os.environ.get(f"{mod.NAME.upper()}_DAILY_CAP", "100" if g._paid_at_import() else "25"))
    state = lambda n: g.ROOT / "cache" / f"{mod.NAME}_{n}.{g._which()}"

    def read_float(p: Path) -> float:
        try:
            return float(p.read_text())
        except (OSError, ValueError):
            return 0.0

    def recent() -> list[float]:
        try:
            return [t for t in json.loads(state("calls").read_text()) if time.time() - t < g.WINDOW_S]
        except (OSError, ValueError):
            return []

    def pace() -> None:
        wait = read_float(state("blocked_until")) - time.time()
        if wait > 0:
            raise QuotaExhausted(f"{mod.MODEL} said to retry in {wait / 3600:.1f} more hours.")
        now = g.datetime.now(g.ZoneInfo("America/Los_Angeles"))
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        if sum(t >= day_start for t in recent()) >= cap:
            raise QuotaExhausted(f"{mod.MODEL} cap of {cap} requests today (Pacific) reached "
                                 f"({mod.NAME.upper()}_DAILY_CAP overrides).")
        wait = g.gap_s() - (time.time() - read_float(state("last_call")))
        if wait > 0:
            print(f"[{mod.NAME}] waiting {wait:.0f}s between calls", flush=True)
            time.sleep(wait)

    def transcribe(wav_path: str) -> str:
        key = g._key()
        body = json.dumps(mod.build_request(wav_path)).encode()
        state("last_call").parent.mkdir(exist_ok=True)
        for attempt in (1, 2):
            state("last_call").write_text(str(time.time()))
            req = urllib.request.Request(f"{API}/v1beta/models/{mod.MODEL}:generateContent", data=body, method="POST",
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.load(r)
                state("calls").write_text(json.dumps(recent() + [time.time()]))
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                if e.code != 429:
                    raise RuntimeError(f"{mod.MODEL} HTTP {e.code}: {detail[:300]}") from None
                retry_after = e.headers.get("Retry-After")
                if re.search(r"per ?day|daily|PerDay", detail, re.I):
                    if retry_after and retry_after.isdigit():
                        state("blocked_until").write_text(str(time.time() + int(retry_after)))
                    raise QuotaExhausted(f"{mod.MODEL} daily limit reached: {detail[:240]}") from None
                if attempt == 2:
                    raise RuntimeError(f"{mod.MODEL} rate-limited again; giving up. {detail[:200]}") from None
                m = re.search(r"(\d+(?:\.\d+)?)s", detail)
                delay = min(float(m.group(1)) + 2 if m else 60.0, g.MAX_RETRY_WAIT_S)
                print(f"[{mod.NAME}] 429, waiting {delay:.0f}s", flush=True)
                time.sleep(delay)
        return mod.parse(response_text(data, mod.MODEL))

    return pace, transcribe
