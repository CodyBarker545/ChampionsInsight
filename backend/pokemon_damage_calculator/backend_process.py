"""Lifecycle helpers for the bundled Smogon JavaScript damage calculator."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


DAMAGE_CALC_DIR = Path(__file__).resolve().parent / "damage-calc-master"
DAMAGE_CALC_PORT = int(os.environ.get("DAMAGE_CALC_PORT", "3001"))
DAMAGE_CALC_URL = f"http://127.0.0.1:{DAMAGE_CALC_PORT}"
_process: subprocess.Popen | None = None
LOG_PATH = Path(__file__).resolve().parents[1] / "test-output" / "damage_calc_backend.log"
_log_handle = None
_startup_lock = threading.Lock()
# This service is local; system proxy settings must not intercept its requests.
LOCAL_HTTP = build_opener(ProxyHandler({}))


def _is_backend_ready() -> bool:
    try:
        with LOCAL_HTTP.open(f"{DAMAGE_CALC_URL}/health", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status == 200 and payload.get("status") == "ok" and payload.get("engine") == "@smogon/calc"
    except (OSError, URLError, ValueError):
        return False


def start_damage_calc_backend(timeout_seconds: float = 60.0) -> None:
    """Starts the bundled Node backend if it is not already reachable."""
    with _startup_lock:
        _start_backend(timeout_seconds)


def _startup_error(message: str) -> RuntimeError:
    output = ""
    try:
        output = LOG_PATH.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
    except OSError:
        pass
    details = f"\nNode output:\n{output}" if output else ""
    return RuntimeError(f"{message} Check {LOG_PATH}.{details}")


def _start_backend(timeout_seconds: float) -> None:
    global _process
    global _log_handle

    if _is_backend_ready():
        return

    if _process is None or _process.poll() is not None:
        env = os.environ.copy()
        env["DAMAGE_CALC_PORT"] = str(DAMAGE_CALC_PORT)
        if _log_handle is not None:
            _log_handle.close()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _log_handle = LOG_PATH.open("wb")
        try:
            _process = subprocess.Popen(
                ["node", "server.js"],
                cwd=DAMAGE_CALC_DIR,
                env=env,
                stdout=_log_handle,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as error:
            _log_handle.close()
            _log_handle = None
            raise RuntimeError(f"Could not launch Node.js for the damage calculator: {error}") from error

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _is_backend_ready():
            return
        if _process.poll() is not None:
            error = _startup_error(f"Smogon damage calculator backend exited with code {_process.returncode} before becoming ready.")
            stop_damage_calc_backend()
            raise error
        time.sleep(0.2)

    error = _startup_error(f"Timed out after {timeout_seconds:g} seconds waiting for the Smogon damage calculator backend at {DAMAGE_CALC_URL}.")
    stop_damage_calc_backend()
    raise error


def stop_damage_calc_backend() -> None:
    """Stops the child Node backend started by this Python process."""
    global _process
    global _log_handle

    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait(timeout=5)
    _process = None
    if _log_handle is not None:
        _log_handle.close()
        _log_handle = None


atexit.register(stop_damage_calc_backend)
