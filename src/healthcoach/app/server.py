"""Tiny local HTTP server (standard library only). Binds to localhost; never logs request bodies.

Defences for a local health app: Host/Origin checks (blocks DNS rebinding and cross-site requests), JSON-only POSTs,
a body size cap, a strict Content-Security-Policy, whitelisted static files, and no caching of API responses."""
from __future__ import annotations
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from .core import AppError, HealthCoach

STATIC = Path(__file__).parent / "static"
FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
         "/static/app.css": ("app.css", "text/css; charset=utf-8")}
MAX_BODY = 64 * 1024
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def make_server(coach: HealthCoach, host: str = "127.0.0.1", port: int = 8888) -> ThreadingHTTPServer:
    assets = {path: ((STATIC / name).read_bytes(), ctype) for path, (name, ctype) in FILES.items()}

    class Handler(BaseHTTPRequestHandler):
        server_version = "HealthCoach"

        def log_message(self, fmt, *args):      # never log: requests may reference a session
            pass

        # ---- helpers -------------------------------------------------------------------------------------------
        def _send(self, status: int, body: bytes, ctype: str, api: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store" if api else "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: dict) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json", api=True)

        def _host_ok(self) -> bool:
            port_ = self.server.server_address[1]
            host_hdr = self.headers.get("Host", "")
            parts = urlsplit("//" + host_hdr)
            if parts.hostname is None or ("[" + parts.hostname + "]" if ":" in parts.hostname else parts.hostname) not in LOCAL_HOSTS:
                return False
            if parts.port not in (None, port_):
                return False
            origin = self.headers.get("Origin")
            return origin is None or urlsplit(origin).netloc == host_hdr

        def _guard(self) -> bool:
            if not self._host_ok():
                self._json(403, {"error": "Forbidden"})
                return False
            return True

        # ---- routes --------------------------------------------------------------------------------------------
        def do_GET(self):
            if not self._guard():
                return
            path = urlsplit(self.path).path
            if path in assets:
                body, ctype = assets[path]
                return self._send(200, body, ctype)
            if path == "/api/health":
                return self._json(200, coach.health())
            self._json(404, {"error": "Not found"})

        def do_POST(self):
            if not self._guard():
                return
            path = urlsplit(self.path).path
            if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
                return self._json(415, {"error": "JSON only"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "Bad request"})
            if n > MAX_BODY:
                return self._json(413, {"error": "Request too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._json(400, {"error": "Bad request"})
            routes = {"/api/start": lambda: coach.start(),
                      "/api/answer": lambda: coach.answer(body.get("sid"), body.get("value")),
                      "/api/back": lambda: coach.back(body.get("sid")),
                      "/api/result": lambda: coach.result(body.get("sid")),
                      "/api/end": lambda: coach.end(body.get("sid"))}
            if path not in routes:
                return self._json(404, {"error": "Not found"})
            try:
                self._json(200, routes[path]())
            except AppError as e:
                self._json(e.status, {"error": e.message})
            except Exception:
                traceback.print_exc(file=sys.stderr)
                self._json(500, {"error": "Something went wrong on this computer. Please try again."})

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv
