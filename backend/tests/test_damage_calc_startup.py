import json

import pytest

from pokemon_damage_calculator import backend_process as backend


class FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout):
        return self.returncode


@pytest.fixture
def isolated_backend(monkeypatch, upload_dir):
    monkeypatch.setattr(backend, "LOG_PATH", upload_dir / "calculator.log")
    monkeypatch.setattr(backend, "_process", None)
    monkeypatch.setattr(backend, "_log_handle", None)
    monkeypatch.setattr(backend.time, "sleep", lambda seconds: None)
    yield
    backend.stop_damage_calc_backend()


def test_reuses_a_healthy_calculator(isolated_backend, monkeypatch):
    monkeypatch.setattr(backend, "_is_backend_ready", lambda: True)
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Should reuse the running calculator"))
    backend.start_damage_calc_backend()
    assert backend._process is None


def test_waits_for_child_to_become_ready(isolated_backend, monkeypatch):
    readiness = iter([False, False, True])
    monkeypatch.setattr(backend, "_is_backend_ready", lambda: next(readiness))
    child = FakeProcess()
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *args, **kwargs: child)
    backend.start_damage_calc_backend()
    assert backend._process is child
    assert not child.terminated


def test_reports_node_output_on_child_exit(isolated_backend, monkeypatch):
    monkeypatch.setattr(backend, "_is_backend_ready", lambda: False)
    child = FakeProcess(returncode=1)

    def launch(*args, **kwargs):
        kwargs["stdout"].write(b"Cannot find module 'express'\n")
        kwargs["stdout"].flush()
        return child

    monkeypatch.setattr(backend.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="Cannot find module 'express'"):
        backend.start_damage_calc_backend()
    assert backend._process is None
    assert backend._log_handle is None


def test_timeout_terminates_owned_child_and_reports_log(isolated_backend, monkeypatch):
    monkeypatch.setattr(backend, "_is_backend_ready", lambda: False)
    child = FakeProcess()
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(RuntimeError, match="Timed out after 0 seconds"):
        backend.start_damage_calc_backend(timeout_seconds=0)
    assert child.terminated
    assert backend._process is None
    assert backend._log_handle is None


def test_reports_missing_node_executable(isolated_backend, monkeypatch):
    monkeypatch.setattr(backend, "_is_backend_ready", lambda: False)

    def launch(*args, **kwargs):
        raise FileNotFoundError("node executable missing")

    monkeypatch.setattr(backend.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="Could not launch Node.js"):
        backend.start_damage_calc_backend()
    assert backend._log_handle is None


def test_health_check_rejects_unrelated_service(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"status": "ok", "engine": "other-service"}).encode()

    monkeypatch.setattr(backend.LOCAL_HTTP, "open", lambda *args, **kwargs: Response())
    assert not backend._is_backend_ready()
