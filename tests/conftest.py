"""Shared pytest fixtures for the ARC-AGI PSO test suite."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from arc.grid import Grid


# ---------------------------------------------------------------------------
# Grid fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_grid() -> Grid:
    """3×3 grid with values 0–8."""
    return np.arange(9, dtype=np.int32).reshape(3, 3)


@pytest.fixture
def identity_task() -> dict:
    """Minimal ARC task where output == input (identity transform)."""
    inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
    return {
        "train": [
            {"input": inp.copy(), "output": inp.copy()},
            {"input": inp * 2 % 10, "output": inp * 2 % 10},
        ],
        "test": [{"input": inp.copy(), "output": inp.copy()}],
    }


@pytest.fixture
def recolor_task() -> dict:
    """Task where every 1 → 2 (simple recolor rule)."""
    def make_pair(inp_data, out_data):
        return {
            "input":  np.array(inp_data, dtype=np.int32),
            "output": np.array(out_data, dtype=np.int32),
        }
    return {
        "train": [
            make_pair([[1, 0], [0, 1]], [[2, 0], [0, 2]]),
            make_pair([[1, 1], [0, 0]], [[2, 2], [0, 0]]),
            make_pair([[0, 0], [1, 0]], [[0, 0], [2, 0]]),
        ],
        "test": [{"input": np.array([[1, 0, 1]], dtype=np.int32),
                  "output": np.array([[2, 0, 2]], dtype=np.int32)}],
    }


@pytest.fixture
def tmp_task_file(identity_task, tmp_path) -> Path:
    """Write identity_task to a temporary JSON file and return its path."""
    raw = {
        "train": [
            {"input": p["input"].tolist(), "output": p["output"].tolist()}
            for p in identity_task["train"]
        ],
        "test": [
            {"input": p["input"].tolist(), "output": p["output"].tolist()}
            for p in identity_task["test"]
        ],
    }
    path = tmp_path / "identity_task.json"
    path.write_text(json.dumps(raw))
    return path


# ---------------------------------------------------------------------------
# LLM / embedding mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """LLMClient mock that returns a valid transform function."""
    llm = MagicMock()
    llm.model = "test-model"
    llm.generate.return_value = (
        "```python\n"
        "def transform(input_grid):\n"
        "    return input_grid.copy()\n"
        "```"
    )
    llm.embed_code.return_value = np.random.rand(768).astype(np.float32)
    return llm


@pytest.fixture
def fixed_embedding() -> np.ndarray:
    """Deterministic unit-norm embedding vector for reproducible tests."""
    rng = np.random.default_rng(42)
    v   = rng.standard_normal(768).astype(np.float32)
    return v / np.linalg.norm(v)
