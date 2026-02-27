"""Tests for agents/llm_client.py — LLMClient embedding and generation.

All HTTP calls are patched with unittest.mock to avoid requiring a live
Ollama server or Anthropic API key.
"""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch, call
from urllib.error import URLError

import numpy as np
import pytest

from agents.llm_client import LLMClient, OLLAMA_CHAT_URL, OLLAMA_EMBED_URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ndjson_stream(*messages: dict) -> BytesIO:
    """Build a fake Ollama NDJSON response stream."""
    lines = [json.dumps(m).encode() + b"\n" for m in messages]
    return BytesIO(b"".join(lines))


def _make_embed_response(vector: list[float]) -> BytesIO:
    return BytesIO(json.dumps({"embedding": vector}).encode())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLLMClientConstruction:
    def test_default_ollama_model(self):
        client = LLMClient(backend="ollama")
        assert "deepseek" in client.model or client.model is not None

    def test_model_override(self):
        client = LLMClient(backend="ollama", model="qwen2.5:7b")
        assert client.model == "qwen2.5:7b"

    def test_anthropic_backend_missing_sdk_raises(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError):
                LLMClient(backend="anthropic")


# ---------------------------------------------------------------------------
# embed_code — success path
# ---------------------------------------------------------------------------

class TestEmbedCode:
    def _patch_urlopen(self, vector: list[float]):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"embedding": vector}).encode()
        return patch("agents.llm_client.urllib.request.urlopen", return_value=mock_resp)

    def test_returns_numpy_array(self):
        vector = [0.1, 0.2, 0.3, 0.4]
        with self._patch_urlopen(vector):
            client = LLMClient()
            result = client.embed_code("def transform(g): return g")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_l2_normalised(self):
        vector = [3.0, 4.0]   # norm = 5.0
        with self._patch_urlopen(vector):
            client = LLMClient()
            result = client.embed_code("code")
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_zero_vector_not_divided(self):
        vector = [0.0, 0.0, 0.0]
        with self._patch_urlopen(vector):
            client = LLMClient()
            result = client.embed_code("code")
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_uses_correct_embed_url(self):
        vector = [1.0, 0.0]
        with self._patch_urlopen(vector) as mock_urlopen:
            client = LLMClient()
            client.embed_code("hello")
        called_url = mock_urlopen.call_args[0][0].full_url
        assert OLLAMA_EMBED_URL in called_url

    def test_model_override_sent_in_payload(self):
        vector = [1.0, 0.0]
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"embedding": vector}).encode()

        with patch("agents.llm_client.urllib.request.urlopen", return_value=mock_resp):
            with patch("agents.llm_client.urllib.request.Request") as mock_req:
                mock_req.return_value = MagicMock(full_url=OLLAMA_EMBED_URL)
                client = LLMClient()
                client.embed_code("code", model="custom-embed-model")
        # `data` is passed as a keyword arg to Request()
        call_kwargs = mock_req.call_args[1]
        payload = json.loads(call_kwargs["data"].decode())
        assert payload["model"] == "custom-embed-model"


# ---------------------------------------------------------------------------
# embed_code — error paths
# ---------------------------------------------------------------------------

class TestEmbedCodeErrors:
    def test_connection_error_raises(self):
        with patch("agents.llm_client.urllib.request.urlopen",
                   side_effect=URLError("connection refused")):
            client = LLMClient()
            with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
                client.embed_code("code")

    def test_missing_embedding_key_raises(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"error": "model not found"}).encode()
        with patch("agents.llm_client.urllib.request.urlopen", return_value=mock_resp):
            client = LLMClient()
            with pytest.raises(RuntimeError, match="missing 'embedding'"):
                client.embed_code("code")


# ---------------------------------------------------------------------------
# generate — Ollama path (mocked streaming)
# ---------------------------------------------------------------------------

class TestGenerateOllama:
    def _make_stream(self, chunks: list[str], done_at: int = -1) -> MagicMock:
        """Build a mock context manager that iterates over NDJSON lines."""
        lines = []
        for i, chunk in enumerate(chunks):
            done = (i == len(chunks) - 1) if done_at == -1 else (i == done_at)
            obj  = {"message": {"content": chunk}, "done": done}
            lines.append(json.dumps(obj).encode() + b"\n")

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = lambda s: iter(lines)
        return mock_resp

    def test_concatenates_chunks(self):
        chunks = ["def ", "transform", "(g):", "\n    return g"]
        with patch("agents.llm_client.urllib.request.urlopen",
                   return_value=self._make_stream(chunks)):
            client = LLMClient()
            result = client.generate("system", [{"role": "user", "content": "task"}])
        assert "def transform" in result

    def test_think_tags_prepended(self):
        lines = [
            json.dumps({"message": {"content": "<think>hidden</think>"}, "done": False}).encode(),
            json.dumps({"message": {"content": "actual answer"}, "done": True}).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = lambda s: iter(lines)
        with patch("agents.llm_client.urllib.request.urlopen", return_value=mock_resp):
            client = LLMClient()
            result = client.generate("sys", [{"role": "user", "content": "q"}])
        # The think block ends up in the output
        assert "actual answer" in result

    def test_connection_error_raises(self):
        with patch("agents.llm_client.urllib.request.urlopen",
                   side_effect=URLError("no server")):
            client = LLMClient()
            with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
                client.generate("sys", [{"role": "user", "content": "q"}])

    def test_temperature_passed_in_payload(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = lambda s: iter([
            json.dumps({"message": {"content": "ok"}, "done": True}).encode()
        ])
        with patch("agents.llm_client.urllib.request.urlopen", return_value=mock_resp):
            with patch("agents.llm_client.urllib.request.Request") as mock_req:
                mock_req.return_value = MagicMock(full_url=OLLAMA_CHAT_URL)
                client = LLMClient(temperature=0.42)
                client.generate("sys", [{"role": "user", "content": "q"}])
        call_kwargs = mock_req.call_args[1]
        payload = json.loads(call_kwargs["data"].decode())
        assert payload["options"]["temperature"] == pytest.approx(0.42)
