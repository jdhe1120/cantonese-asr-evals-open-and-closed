"""MAI-Transcribe-2 through the Azure Speech fast-transcription REST API (Microsoft Foundry).

Documented at learn.microsoft.com/azure/ai-services/speech-service/mai-transcribe (public preview):
  POST https://<resource>.cognitiveservices.azure.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15
  header Ocp-Apim-Subscription-Key, multipart form: `audio` (the file) + `definition` (JSON).
No Foundry model deployment is needed; the model is picked by enhancedMode.model.

Settings come from .env (gitignored), never from the code:
  MAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com
  MAI_API_KEY=<key of that Speech/Foundry resource>

Region: the file API with MAI-Transcribe is only available in centralindia, eastus, northeurope,
southeastasia, westus and westus2 (2026-10-08). The key is region-scoped.

Language is forced to Cantonese (`yue`), as every other model here is told the audio is Cantonese;
transcribeStyle is verbatim (the default for MAI-Transcribe-2), like the Gemini adapter.
"""
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
API_VERSION = "2025-10-15"
MIN_GAP_S = 1.0  # rate limits are unpublished; keep calls polite
_last_call = 0.0


def _env() -> dict[str, str]:
    out = {}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _multipart(fields: list[tuple[str, str, bytes, str | None]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = b""
    for name, filename, data, ctype in fields:
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        body += f"--{boundary}\r\nContent-Disposition: {disp}\r\n".encode()
        if ctype:
            body += f"Content-Type: {ctype}\r\n".encode()
        body += b"\r\n" + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _text(resp: dict) -> str:
    combined = resp.get("combinedPhrases")
    if combined:
        return "".join(p.get("text", "") for p in combined).strip()
    phrases = resp.get("phrases")
    if phrases:
        return "".join(p.get("text", "") for p in phrases).strip()
    raise RuntimeError(f"MAI response had neither combinedPhrases nor phrases: {json.dumps(resp)[:300]}")


def transcribe(wav_path: str) -> str:
    global _last_call
    env = _env()
    endpoint, key = env.get("MAI_ENDPOINT", "").rstrip("/"), env.get("MAI_API_KEY", "")
    if not endpoint or not key or "<" in endpoint or "<" in key:  # blank, or still the <...> filler
        raise RuntimeError("MAI_ENDPOINT and MAI_API_KEY are missing from .env")
    definition = json.dumps({"locales": ["yue"], "enhancedMode": {"enabled": True, "model": "MAI-Transcribe-2"}})
    body, ctype = _multipart([
        ("audio", Path(wav_path).name, Path(wav_path).read_bytes(), "audio/wav"),
        ("definition", "", definition.encode(), "application/json"),
    ])
    url = f"{endpoint}/speechtotext/transcriptions:transcribe?api-version={API_VERSION}"
    for attempt in (1, 2):
        wait = MIN_GAP_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": ctype, "Ocp-Apim-Subscription-Key": key})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return _text(json.load(r))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            retry = e.headers.get("Retry-After")
            if e.code == 429 and attempt == 1 and retry and retry.isdigit() and int(retry) <= 120:
                print(f"[mai] 429, waiting {retry}s as Azure asked", flush=True)
                time.sleep(int(retry))
                continue
            raise RuntimeError(f"MAI HTTP {e.code}: {detail}") from None
    raise RuntimeError("unreachable")
