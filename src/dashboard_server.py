"""Minimal local web server for the live dashboard (standard library only)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import config
from src.dashboard_state import DashboardState
from src.utils import get_logger

logger = get_logger("dashboard_server")
_MAX_BODY_BYTES = 1024
_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")


def validate_run_params(payload: dict) -> tuple[float, float]:
    """Check requested run length and speed against ``config`` limits.

    Raises:
        ValueError: with a user-readable message if a value is missing or out of range.
    """
    try:
        hours, speed = float(payload["hours"]), float(payload["speed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Provide numeric 'hours' and 'speed'.") from exc
    low_h, high_h = config.DASHBOARD_HOURS_RANGE
    low_s, high_s = config.DASHBOARD_SPEED_RANGE
    if not (low_h <= hours <= high_h):
        raise ValueError(f"hours must be between {low_h} and {high_h}.")
    if not (low_s <= speed <= high_s):
        raise ValueError(f"speed must be between {low_s} and {high_s}.")
    return hours, speed


def make_handler(state: DashboardState, restart: Callable[[float, float], None],
                 page: Path = config.DASHBOARD_PAGE) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to a state object and a restart callback."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "ThreatDashboard/1.0"

        def log_message(self, fmt: str, *args) -> None:  # keep the console clean
            logger.debug(fmt, *args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", _CSP)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            route = self.path.split("?", 1)[0]
            if route in ("/", "/index.html"):
                if not page.exists():
                    self._json(500, {"error": f"Dashboard page not found: {page}"})
                    return
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/state":
                self._json(200, state.snapshot())
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/restart":
                self._json(404, {"error": "Not found"})
                return
            # Browsers cannot add this header cross-origin without a CORS preflight,
            # which blocks other websites from triggering restarts.
            if self.headers.get("X-Requested-With") != "dashboard":
                self._json(403, {"error": "Missing X-Requested-With header."})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > _MAX_BODY_BYTES:
                self._json(400, {"error": "Request body must be small JSON."})
                return
            try:
                hours, speed = validate_run_params(json.loads(self.rfile.read(length)))
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            restart(hours, speed)
            self._json(200, {"ok": True, "hours": hours, "speed": speed})

    return Handler


class DashboardServer:
    """Wraps ``ThreadingHTTPServer`` with the dashboard routes."""

    def __init__(self, state: DashboardState, restart: Callable[[float, float], None],
                 host: str = config.DASHBOARD_HOST, port: int = config.DASHBOARD_PORT) -> None:
        self.httpd = ThreadingHTTPServer((host, port), make_handler(state, restart))
        self.httpd.daemon_threads = True

    @property
    def address(self) -> tuple[str, int]:
        """(host, port) actually bound (useful when port 0 was requested)."""
        return self.httpd.server_address[0], self.httpd.server_address[1]

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def close(self) -> None:
        """Stop serving and release the port."""
        self.httpd.shutdown()
        self.httpd.server_close()
