#!/usr/bin/env python3
"""ComfyRanch worker status server.

Read-only progress channel that runs inside the rented GPU container alongside
ComfyUI. The batch itself is baked into the deploy env (``COMFRANCH_TASKS_B64``)
and written to ``tasks.json`` by the bootstrap, so there is **no write path
here** — the app only reads progress and harvests outputs from ComfyUI.

Endpoints (CORS-enabled so the app can call cross-origin):

    GET  /healthz   -> 200 {"ok": true}
    GET  /status    -> the runner's status.json (+ done flag)
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("comfyranch.status")

DEFAULT_AUTOMATION_DIR = "/workspace/automation"


class StatusConfig:
    def __init__(self, automation_dir: Path) -> None:
        self.status_path = automation_dir / "status.json"
        self.done_path = automation_dir / "DONE"


class _Handler(BaseHTTPRequestHandler):
    server_version = "ComfyRanchStatus/0.1"
    protocol_version = "HTTP/1.1"
    config: StatusConfig  # set on the server class before serving

    # -- helpers ----------------------------------------------------------- #
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # route to our logger
        log.debug("%s - %s", self.address_string(), fmt % args)

    # -- routes ------------------------------------------------------------ #
    def do_OPTIONS(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            self._send_json(200, {"ok": True})
            return
        if path == "/status":
            try:
                payload = json.loads(self.config.status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {"state": "waiting", "total": 0, "completed": 0, "failed": 0}
            payload["done"] = self.config.done_path.is_file()
            self._send_json(200, payload)
            return
        self._send_json(404, {"error": "not found"})


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, config: StatusConfig) -> None:
        super().__init__(addr, handler)
        self.config = config
        # Handlers are constructed per-request by the server, so the config must
        # live on the handler class (not an instance) for self.config to resolve.
        handler.config = config


def start_control_server(
    port: int,
    automation_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[ThreadingHTTPServer]:
    """Start the status server in a daemon thread. Returns the server (or None)."""
    (logger or log).info("starting status server on 0.0.0.0:%s", port)
    config = StatusConfig(automation_dir=automation_dir)
    try:
        server = _Server(("0.0.0.0", port), _Handler, config)
    except OSError as exc:
        (logger or log).error("could not bind status server on port %s: %s", port, exc)
        return None
    thread = threading.Thread(target=server.serve_forever, name="comfyranch-status", daemon=True)
    thread.start()
    return server
