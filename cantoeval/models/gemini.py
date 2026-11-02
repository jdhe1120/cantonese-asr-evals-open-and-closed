"""Gemini 3.5 Transcribe over REST.

Uses the dev free-tier key, GEMINI_API_FREE_KEY in .env, unless a billing-enabled key is set as GEMINI_API
(or GEMINI_API_KEY), which then takes over and skips the free-tier cap and gap below.

Free-tier limit, from Google's own 429 body (2026-10-08): "limit: 25 requests per day
on Free Tier" for gemini-3.5-transcribe; Tier 1 (paid) is 100 a day. Quotas reset at midnight Pacific: on
2026-10-23 a 23:25 PDT refusal said "retry in 34m58s" (earlier blocked-until times had landed on 16:59:59
Pacific, i.e. midnight UTC, so Google may have moved it). Everything here is defensive:
  * at least MIN_GAP_S between any two calls, remembered across runs in
    cache/gemini_last_call, so separate invocations still respect it;
  * at most DAILY_CAP (25) successful calls since the last midnight Pacific, remembered in
    cache/gemini_calls.json;
  * on a 429 we record Google's Retry-After in cache/gemini_blocked_until and refuse to
    call again until then. A per-day 429 stops Gemini for the rest of the run
    (QuotaExhausted; finished clips stay cached, so run again later); any other 429
    waits Google's retry delay once, then gives up.
The hypothesis cache means a clip is never sent twice.
"""
import base64
import json
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "gemini-3.5-transcribe"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
MIN_GAP_S = 20.0       # our guess, ~3 requests/min: deliberately below any plausible free-tier RPM
MAX_RETRY_WAIT_S = 120.0
def _state(name: str) -> Path:
    """Per-key state file, so the paid and free keys don't share a "blocked until" time or a call count.
    GEMINI_USE=free forces the free dev key even when a paid one is set."""
    return ROOT / "cache" / f"gemini_{name}.{_which()}"


def _paid_at_import() -> bool:
    env = ROOT / ".env"
    return os.environ.get("GEMINI_USE") != "free" and env.exists() and any(l.startswith(("GEMINI_API=", "GEMINI_API_KEY=")) and l.split("=", 1)[1].strip()
                                for l in env.read_text().splitlines())


DAILY_CAP = int(os.environ.get("GEMINI_DAILY_CAP", "100" if _paid_at_import() else "25"))
WINDOW_S = 24 * 3600  # how long call stamps are kept in the state file


class QuotaExhausted(RuntimeError):
    halt_model = True  # the runner stops calling this model for the rest of the run


def _recent_calls() -> list[float]:
    try:
        stamps = json.loads(_state("calls").read_text())
    except (OSError, ValueError):
        return []
    return [t for t in stamps if time.time() - t < WINDOW_S]


def _calls_today() -> int:
    """Calls since the last midnight Pacific, when Google resets the daily quota (its 429 on 2026-10-23 said
    "retry in 34m58s" at 23:25 PDT)."""
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return sum(t >= day_start for t in _recent_calls())


def _count_call() -> None:
    _state("calls").parent.mkdir(exist_ok=True)
    _state("calls").write_text(json.dumps(_recent_calls() + [time.time()]))


def _blocked_for() -> float:
    try:
        return float(_state("blocked_until").read_text()) - time.time()
    except (OSError, ValueError):
        return 0.0


def _env_value(name: str) -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    return ""


def _which() -> str:
    """Which key this run uses: "free" (GEMINI_USE=free, or no paid key), else "paid"."""
    use = os.environ.get("GEMINI_USE")
    if use == "free" or not (_env_value("GEMINI_API") or _env_value("GEMINI_API_KEY")):
        return "free"
    return "paid"


def _key() -> str:
    """The billing-enabled key if one is set (GEMINI_API, or GEMINI_API_KEY), else the free-tier dev key."""
    which = _which()
    if which == "paid":
        return _env_value("GEMINI_API") or _env_value("GEMINI_API_KEY")
    return _env_value("GEMINI_API_FREE_KEY") or _fail()


def _paid() -> bool:
    return _which() != "free"


def _fail() -> str:
    raise RuntimeError("no Gemini key in .env (GEMINI_API for a paid key, or GEMINI_API_FREE_KEY)")


PAID_GAP_S = 5.0  # paid key: no 25/day cap; a generous gap keeps requests per minute low and spreads the calls out


def gap_s() -> float:
    return PAID_GAP_S if _paid() else MIN_GAP_S


def pace() -> None:
    """Called by the runner before the timer starts. Refuse while Google has told us to wait (both key types), then wait
    out the gap. Free key: also a 25-per-24h cap. Paid key (GEMINI_API in .env): Tier 1 allows 100 requests a day for
    this model (Google's own 429 on 2026-10-08), so the cap there is 100 unless GEMINI_DAILY_CAP says otherwise."""
    wait = _blocked_for()
    if wait > 0:
        raise QuotaExhausted(f"Gemini said to retry in {wait / 3600:.1f} more hours (daily request limit reached).")
    if _calls_today() >= DAILY_CAP:
        raise QuotaExhausted(f"Gemini cap of {DAILY_CAP} requests today (Pacific) reached; run again after midnight Pacific "
                             "(GEMINI_DAILY_CAP overrides).")
    gap = PAID_GAP_S if _paid() else MIN_GAP_S
    wait = gap - (time.time() - _last_call_time())
    if wait > 0:
        print(f"[gemini] waiting {wait:.0f}s between calls", flush=True)
        time.sleep(wait)


def _wait_for_gap() -> None:
    try:
        last = float(_state("last_call").read_text())
    except (OSError, ValueError):
        last = 0.0
    wait = MIN_GAP_S - (time.time() - last)
    if wait > 0:
        print(f"[gemini] waiting {wait:.0f}s to respect the free-tier gap", flush=True)
        time.sleep(wait)


def _last_call_time() -> float:
    try:
        return float(_state("last_call").read_text())
    except (OSError, ValueError):
        return 0.0


def _mark_call() -> None:
    _state("last_call").parent.mkdir(exist_ok=True)
    _state("last_call").write_text(str(time.time()))


def _post(body: bytes, key: str) -> dict:
    req = urllib.request.Request(ENDPOINT, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def transcribe(wav_path: str) -> str:
    key = _key()
    audio = base64.b64encode(Path(wav_path).read_bytes()).decode()
    body = json.dumps({
        "model": MODEL,
        "input": [{"type": "audio", "data": audio, "mime_type": "audio/wav"}],
        "generation_config": {"transcription_config": {
            "language_codes": ["yue-Hant-HK"], "mode": {"type": "verbatim"}}},
    }).encode()
    for attempt in (1, 2):
        if attempt == 2:
            _wait_for_gap()
        _mark_call()
        try:
            data = _post(body, key)
            _count_call()  # only requests Google accepted count toward our own cap
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            if e.code != 429:
                raise RuntimeError(f"Gemini HTTP {e.code}: {detail[:300]}") from None
            retry_after = e.headers.get("Retry-After")
            if re.search(r"per ?day|daily|PerDay", detail, re.I):
                if retry_after and retry_after.isdigit():
                    _state("blocked_until").write_text(str(time.time() + int(retry_after)))
                raise QuotaExhausted(f"Gemini daily request limit reached: {detail[:240]}") from None
            m = re.search(r'retry(?:Delay)?"?\s*[:=]?\s*"?(\d+(?:\.\d+)?)s', detail) or \
                re.search(r"retry in (\d+(?:\.\d+)?)s", detail, re.I)
            delay = min(float(m.group(1)) + 2 if m else 60.0, MAX_RETRY_WAIT_S)
            if attempt == 2:
                raise RuntimeError(f"Gemini rate-limited again after waiting; giving up. {detail[:200]}") from None
            print(f"[gemini] 429, waiting {delay:.0f}s as Google asked", flush=True)
            time.sleep(delay)
    if data.get("error"):
        raise RuntimeError(f"Gemini: {data['error'].get('message')}")
    if data.get("status") not in (None, "completed"):
        raise RuntimeError(f"Gemini status: {data.get('status')}")
    return "".join(c.get("text", "") for s in data.get("steps", []) if s.get("type") == "model_output"
                   for c in s.get("content", []) if c.get("type") == "text").strip()
