"""Loads the local language model used to write RAG answers."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATION_MODEL_CACHE_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
)


def get_local_generation_model_path() -> Path | None:
    """Return the newest downloaded generation model path, if it exists."""

    if not GENERATION_MODEL_CACHE_DIR.exists():
        return None

    snapshots = sorted(
        [path for path in GENERATION_MODEL_CACHE_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def load_generation_model():
    """Load the local LLM used to write final answers from retrieved context."""

    model_path = get_local_generation_model_path()
    model_name_or_path = str(model_path) if model_path is not None else GENERATION_MODEL_NAME
    local_only = model_path is not None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.float32,
        device_map=None,
        local_files_only=local_only,
    )
    model.eval()
    return tokenizer, model
