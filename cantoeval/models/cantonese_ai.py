"""cantonese.ai speech-to-text over its synchronous REST API (closed source).

Documented at docs.cantonese.ai/speech-to-text (2026-10-19):
  POST https://cantonese.ai/api/stt, multipart form: `api_key`, `data` (the file), options.
`wait_for_completion=true` is required here, or the endpoint answers with just {request_id, status}.
Fusion (second-pass jyutping ASR + LLM homophone correction) is off by default; we leave it off so this is the
raw recognizer, like every other row.

Key: CANTONESE_AI_API_KEY in .env (gitignored).

Paid plans: set CANTONESE_AI_RPM (see gap_s); they have no daily request cap, only hours of audio a month.

Free tier (no speech-to-text plan): 10 accepted requests per UTC day, audio up to 5 minutes, ~3 requests/minute
(pricing page). A 429 `free_daily_limit` stops this model for the rest of the run and its Retry-After is kept in
cache/cantonese_ai_blocked_until, so later runs don't call before 00:00 UTC. Rejected requests (too long, invalid)
don't count. The hypothesis cache means a clip is never sent twice.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .mai import _env, _multipart

ROOT = Path(__file__).resolve().parent.parent.parent
ENDPOINT = "https://cantonese.ai/api/stt"
# Cloudflare in front of the API answers 403 "error code: 1010" to urllib's default Python-urllib User-Agent.
USER_AGENT = "cantoeval/0.1 (+https://cantonese.ai/api/stt client)"
MIN_GAP_S = 30.0  # pricing page says 3/min on the free tier, but 2026-10-20 a 20 s gap got every third call a 429
MINUTE_RETRY_S = 65.0  # a per-minute 429 ({"message":"Too many requests"}) is retried once after this wait
BLOCKED = ROOT / "cache" / "cantonese_ai_blocked_until"
LAST_CALL = ROOT / "cache" / "cantonese_ai_last_call"
RAW_DIR = ROOT / "cache" / "cantonese_ai_raw"  # every raw response, kept before parsing


class QuotaExhausted(RuntimeError):
    halt_model = True  # the runner stops calling this model for the rest of the run


def _read_float(p: Path) -> float:
    try:
        return float(p.read_text())
    except (OSError, ValueError):
        return 0.0


def gap_s() -> float:
    """Seconds between calls: 30 on the free tier; with a paid plan, set CANTONESE_AI_RPM in .env to its
    speech-to-text requests per minute (Lite 5, Pro 10, Ultra 20) and the gap is 60/RPM plus a second of margin."""
    rpm = _env().get("CANTONESE_AI_RPM", "").strip()
    return 60.0 / float(rpm) + 1.0 if rpm else MIN_GAP_S


def pace() -> None:
    """Called by the runner before the timer starts: refuse while blocked, then wait out the per-minute gap."""
    wait = _read_float(BLOCKED) - time.time()
    if wait > 0 and not _env().get("CANTONESE_AI_RPM", "").strip():  # a paid plan has no daily cap: ignore free-tier blocks
        raise QuotaExhausted(f"cantonese.ai daily limit reached; resets in {wait / 3600:.1f} h (00:00 UTC).")
    wait = gap_s() - (time.time() - _read_float(LAST_CALL))
    if wait > 0:
        print(f"[cantonese-ai] waiting {wait:.0f}s between calls", flush=True)
        time.sleep(wait)


def transcribe(wav_path: str) -> str:
    key = _env().get("CANTONESE_AI_API_KEY") or ""
    if not key:
        raise RuntimeError("no CANTONESE_AI_API_KEY in .env")
    body, ctype = _multipart([
        ("api_key", "", key.encode(), None),
        ("wait_for_completion", "", b"true", None),
        ("with_timestamp", "", b"false", None),
        ("with_diarization", "", b"false", None),
        ("data", Path(wav_path).name, Path(wav_path).read_bytes(), "audio/wav"),
    ])
    req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={"Content-Type": ctype, "User-Agent": USER_AGENT})
    LAST_CALL.parent.mkdir(exist_ok=True)
    for attempt in (1, 2):
        LAST_CALL.write_text(str(time.time()))
        try:
            with urllib.request.urlopen(req, timeout=110) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            retry_after = e.headers.get("Retry-After")
            if e.code == 429 and "free_daily_limit" in detail:
                BLOCKED.write_text(str(time.time() + (int(retry_after) if retry_after and retry_after.isdigit() else 3600)))
                raise QuotaExhausted("cantonese.ai free tier: 10 requests today (UTC) used; run again after 00:00 UTC.") from None
            if e.code == 429 and attempt == 1:  # per-minute limit; rejected requests don't count against the day
                delay = float(retry_after) + 1 if retry_after and retry_after.isdigit() else MINUTE_RETRY_S
                print(f"[cantonese-ai] 429 (per-minute), waiting {delay:.0f}s and retrying once", flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"cantonese.ai HTTP {e.code}: {detail[:300]}") from None
    # Every accepted request counts against the free tier, so keep the raw answer before parsing it.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{Path(wav_path).stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return _text(data)


def _text(data: dict) -> str:
    """The live API (2026-10-20) nests the result: {"success": true, "data": {"transcription": ..., "status":
    "completed", ...}}. The docs show a flat {"text": ...}; accept both."""
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    if inner.get("status") not in (None, "completed"):
        raise RuntimeError(f"cantonese.ai status: {inner.get('status')}")
    for key in ("transcription", "text"):
        if isinstance(inner.get(key), str):
            return inner[key].strip()
    raise RuntimeError(f"cantonese.ai response had no transcription: {json.dumps(data, ensure_ascii=False)[:300]}")
