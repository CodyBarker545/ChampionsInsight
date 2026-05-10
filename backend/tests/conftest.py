"""Provides shared pytest fixtures for backend tests."""

import sys
import shutil
from uuid import uuid4
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture
def client():
    from app import app

    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture
def upload_dir():
    path = BACKEND_DIR / "test-output" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
