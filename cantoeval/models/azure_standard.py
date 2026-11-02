"""Azure Speech standard recognizer (fast transcription, no enhancedMode), locale zh-HK (Cantonese, Traditional).

Same endpoint, key and request as the MAI adapter (.env: MAI_ENDPOINT, MAI_API_KEY); the only difference is the
definition has no `enhancedMode`, so the service uses its regular locale-specific speech models instead of MAI.
Microsoft doesn't name the model; the newest zh-HK base model listed on the resource (2026-10-08) is 20260811, but
which one fast transcription uses is not documented.

On 2026-10-08 a burst of calls got `429 TooManyRequests / Resource Exhausted`, so calls are spaced MIN_GAP_S apart
(pace() runs outside the timer), a 429 waits Retry-After (default 30 s) up to 3 attempts, and if it is still
rate-limited the model is stopped for the rest of the run instead of failing every remaining clip.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import mai

MIN_GAP_S = 3.0
_last_call = 0.0


class RateLimited(RuntimeError):
    halt_model = True  # the runner stops calling this model for the rest of the run


def pace() -> None:
    wait = MIN_GAP_S - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)


def transcribe(wav_path: str) -> str:
    global _last_call
    env = mai._env()
    endpoint, key = env.get("MAI_ENDPOINT", "").rstrip("/"), env.get("MAI_API_KEY", "")
    if not endpoint or not key or "<" in endpoint or "<" in key:
        raise RuntimeError("MAI_ENDPOINT and MAI_API_KEY are missing from .env")
    body, ctype = mai._multipart([
        ("audio", Path(wav_path).name, Path(wav_path).read_bytes(), "audio/wav"),
        ("definition", "", json.dumps({"locales": ["zh-HK"]}).encode(), "application/json"),
    ])
    url = f"{endpoint}/speechtotext/transcriptions:transcribe?api-version={mai.API_VERSION}"
    for attempt in (1, 2, 3):
        _last_call = time.time()
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": ctype, "Ocp-Apim-Subscription-Key": key})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                _last_call = time.time()
                return mai._text(json.load(r))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code != 429:
                raise RuntimeError(f"Azure standard HTTP {e.code}: {detail}") from None
            if attempt == 3:
                raise RateLimited(f"Azure still rate-limiting after 3 tries: {detail}") from None
            retry = e.headers.get("Retry-After")
            delay = int(retry) if retry and retry.isdigit() else 30
            print(f"[azure-standard] 429, waiting {delay}s", flush=True)
            time.sleep(min(delay, 120))
    raise RuntimeError("unreachable")
