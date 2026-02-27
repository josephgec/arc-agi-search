"""LLM backend abstraction for the ARC-AGI multi-agent framework.

Any agent instantiates LLMClient and calls generate() to get a completion.
All network and SDK details are hidden behind that single method.

Supported backends
------------------
  "ollama"    — local Ollama server via urllib streaming NDJSON (no API cost)
  "anthropic" — Anthropic cloud API via the `anthropic` SDK

Embedding
---------
embed_code() uses the Ollama embeddings endpoint (nomic-embed-text by default)
and returns an L2-normalised numpy vector for use in the PSO vector space.
The Anthropic backend falls back to Ollama for embeddings since Anthropic does
not expose a dedicated embeddings API.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Connection constants
# ---------------------------------------------------------------------------

OLLAMA_CHAT_URL       = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL      = "http://localhost:11434/api/embeddings"
DEFAULT_OLLAMA_MODEL  = "deepseek-r1:32b"
DEFAULT_EMBED_MODEL   = "nomic-embed-text"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin, backend-agnostic wrapper around LLM completion and embedding APIs."""

    def __init__(
        self,
        backend:       str        = "ollama",
        model:         str | None = None,
        temperature:   float      = 0.6,
        max_tokens:    int        = 8192,
        timeout:       float      = 120.0,
        embed_timeout: float      = 30.0,
        debug:         bool       = False,
    ) -> None:
        self.backend       = backend
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.timeout       = timeout
        self.embed_timeout = embed_timeout
        self.debug         = debug

        if backend == "anthropic":
            try:
                import anthropic as _anthropic
                self._anthropic_client = _anthropic.Anthropic()
            except ImportError as exc:
                raise ImportError(
                    "Install the 'anthropic' package: pip install anthropic"
                ) from exc
            self.model = model or DEFAULT_ANTHROPIC_MODEL
        else:
            self._anthropic_client = None
            self.model = model or DEFAULT_OLLAMA_MODEL

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        system:      str,
        messages:    list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        """Return a completion string for the given system prompt + messages."""
        temp = temperature if temperature is not None else self.temperature

        if self.backend == "anthropic":
            return self._generate_anthropic(system, messages, temp)
        return self._generate_ollama(system, messages, temp)

    def _generate_ollama(
        self,
        system:      str,
        messages:    list[dict[str, str]],
        temperature: float,
    ) -> str:
        payload = {
            "model":    self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream":   True,
            "options":  {"temperature": temperature},
        }
        data    = json.dumps(payload).encode()
        request = urllib.request.Request(
            OLLAMA_CHAT_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        content   = []
        thinking  = []
        in_think  = False

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = obj.get("message", {})

                    # Handle explicit thinking field (deepseek-r1 style)
                    think_chunk = msg.get("thinking", "")
                    if think_chunk:
                        thinking.append(think_chunk)

                    chunk = msg.get("content", "")
                    if not chunk:
                        continue

                    # Handle inline <think> tags
                    if "<think>" in chunk:
                        in_think = True
                    if in_think:
                        thinking.append(chunk)
                        if "</think>" in chunk:
                            in_think = False
                    else:
                        content.append(chunk)

                    if obj.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_CHAT_URL}: {exc}"
            ) from exc

        text = "".join(content)
        if thinking:
            think_text = "".join(thinking)
            if "<think>" not in think_text:
                think_text = f"<think>{think_text}</think>"
            text = think_text + text

        if self.debug:
            print(f"[llm] ollama/{self.model}: {len(text)} chars")

        return text

    def _generate_anthropic(
        self,
        system:      str,
        messages:    list[dict[str, str]],
        temperature: float,
    ) -> str:
        response = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            temperature=temperature,
        )
        text = response.content[0].text if response.content else ""
        if self.debug:
            print(f"[llm] anthropic/{self.model}: {len(text)} chars")
        return text

    # ------------------------------------------------------------------
    # Embeddings (always via Ollama, model-independent of backend)
    # ------------------------------------------------------------------

    def embed_code(
        self,
        code_str: str,
        model:    str | None = None,
    ) -> np.ndarray:
        """Embed a code string and return an L2-normalised float32 vector.

        Uses the Ollama /api/embeddings endpoint regardless of the chat
        backend.  Default model is nomic-embed-text (768 dimensions).

        Args:
            code_str: The Python source code to embed.
            model:    Override the embedding model (default: nomic-embed-text).

        Returns:
            L2-normalised numpy float32 array of shape (d,).

        Raises:
            ConnectionError: If the Ollama server is unreachable.
            RuntimeError:    If the server returns an unexpected response.
        """
        embed_model = model or DEFAULT_EMBED_MODEL
        payload = json.dumps({"model": embed_model, "prompt": code_str}).encode()
        request = urllib.request.Request(
            OLLAMA_EMBED_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.embed_timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama embedding endpoint at {OLLAMA_EMBED_URL}: {exc}"
            ) from exc

        raw = body.get("embedding")
        if raw is None:
            raise RuntimeError(
                f"Ollama embed response missing 'embedding' key. Got: {list(body.keys())}"
            )

        vec  = np.array(raw, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
