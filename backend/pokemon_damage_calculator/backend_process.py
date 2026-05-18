"""Lifecycle helpers for the bundled Smogon JavaScript damage calculator."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DAMAGE_CALC_DIR = Path(__file__).resolve().parent / "damage-calc-master"
DAMAGE_CALC_PORT = int(os.environ.get("DAMAGE_CALC_PORT", "3001"))
DAMAGE_CALC_URL = f"http://127.0.0.1:{DAMAGE_CALC_PORT}"
_process: subprocess.Popen | None = None


def _is_backend_ready() -> bool:
    try:
        with urlopen(f"{DAMAGE_CALC_URL}/health", timeout=0.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status == 200 and payload.get("status") == "ok"
    except (OSError, URLError, ValueError):
        return False


def start_damage_calc_backend(timeout_seconds: float = 20.0) -> None:
    """Starts the bundled Node backend if it is not already reachable."""
    global _process

    if _is_backend_ready():
        return

    if _process is None or _process.poll() is not None:
        env = os.environ.copy()
        env["DAMAGE_CALC_PORT"] = str(DAMAGE_CALC_PORT)
        _process = subprocess.Popen(
            ["node", "server.js"],
            cwd=DAMAGE_CALC_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _is_backend_ready():
            return
        if _process.poll() is not None:
            raise RuntimeError("Smogon damage calculator backend exited before becoming ready.")
        time.sleep(0.2)

    raise RuntimeError("Timed out waiting for the Smogon damage calculator backend.")


def stop_damage_calc_backend() -> None:
    """Stops the child Node backend started by this Python process."""
    global _process

    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()
    _process = None


atexit.register(stop_damage_calc_backend)
