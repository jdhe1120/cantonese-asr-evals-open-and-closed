"""Tiny local server for the progress UI: / page, /latest.json, /clips/<id>.wav.

/latest.json is runs/latest.json when a run exists; otherwise (a fresh clone) the run rebuilt from results/per_clip.csv.
"""
import functools
import http.server
import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@functools.cache
def _snapshot() -> bytes | None:
    from .snapshot import from_results
    doc = from_results()
    return None if doc is None else json.dumps(doc, ensure_ascii=False).encode()


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/latest.json" and not (ROOT / "runs" / "latest.json").exists():
            body = _snapshot()
            if body is None:
                self.send_error(404, "no run yet: run `python -m cantoeval run`, or keep results/per_clip.csv")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    ROUTES = {"/": "ui/index.html", "/latest.json": "runs/latest.json"}

    def translate_path(self, path: str) -> str:
        path = path.split("?")[0]
        if path in self.ROUTES:
            return str(ROOT / self.ROUTES[path])
        if path.startswith("/clips/") and path.endswith(".wav") and ".." not in path:
            return str(ROOT / path.lstrip("/"))
        return str(ROOT / "ui" / "404")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args) -> None:
        pass


def serve(port: int = 8765, background: bool = False) -> http.server.ThreadingHTTPServer:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if not (ROOT / "runs" / "latest.json").exists():
        threading.Thread(target=_snapshot, daemon=True).start()  # rebuild the diffs before the page asks
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
