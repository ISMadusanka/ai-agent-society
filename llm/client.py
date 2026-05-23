"""
Ollama HTTP API abstraction layer.

Provides a simple `.call(prompt)` → dict interface so that the rest of
the application never deals with HTTP details, retries, or parsing.

Usage::

    client = OllamaClient(base_url="http://localhost:11434", model="gpt-oss:20b")
    result = client.call("Why is the sky blue?")
    print(result["response"])

    structured = client.call_json("List 3 colors as JSON array")
    print(structured)  # Already a Python dict / list
"""

from __future__ import annotations

import json
import time
import logging
from typing import Optional

import requests

from utils.logger import LLM

log = logging.getLogger(__name__)


class OllamaClient:
    """Thin wrapper around the Ollama ``/api/generate`` endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gpt-oss:20b",
        timeout: int = 120,
        max_retries: int = 3,
        temperature: float = 0.8,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self._generate_url = f"{self.base_url}/api/generate"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Send a prompt and return the full Ollama response dict.

        The dict contains at minimum a ``"response"`` key with the
        generated text.  Other metadata (``eval_count``, ``total_duration``,
        etc.) is preserved for callers that need it.
        """
        payload = self._build_payload(prompt, system, temperature)
        return self._post(payload)

    def call_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Send a prompt, force JSON output, and parse it.

        Returns the *parsed* JSON object (dict or list) rather than
        the raw Ollama wrapper.
        """
        payload = self._build_payload(prompt, system, temperature)
        # Do not use payload["format"] = "json" as it causes empty responses
        # in some models (like gpt-oss:20b). We rely on prompt instructions instead.
        raw = self._post(payload)
        text = raw.get("response", "").strip()
        
        # Extract JSON block if wrapped in markdown or surrounded by preamble
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            text = text[start:end+1]
            
        try:
            return json.loads(text)
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning(
                f"{LLM} JSON parse failed. Raw text: {raw.get('response', '')[:100]!r}... Error: {exc}"
            )
            return {"raw": raw.get("response", "")}

    def get_response_text(self, result: dict) -> str:
        """Extract the generated text from an Ollama response dict."""
        return result.get("response", "")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        prompt: str,
        system: Optional[str],
        temperature: Optional[float],
    ) -> dict:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload: dict = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": temperature or self.temperature},
        }
        return payload

    def _post(self, payload: dict) -> dict:
        """POST with retry + exponential backoff."""
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                log.debug(
                    f"{LLM} POST {self._generate_url} "
                    f"(attempt {attempt}/{self.max_retries}, "
                    f"model={self.model})"
                )
                resp = requests.post(
                    self._generate_url,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                log.debug(
                    f"{LLM} Response received "
                    f"({len(data.get('response', ''))} chars)"
                )
                return data

            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                log.warning(
                    f"{LLM} Connection failed (attempt {attempt}). "
                    f"Is Ollama running at {self.base_url}?"
                )
            except requests.exceptions.Timeout as exc:
                last_exc = exc
                log.warning(
                    f"{LLM} Request timed out after {self.timeout}s "
                    f"(attempt {attempt})"
                )
            except requests.exceptions.HTTPError as exc:
                last_exc = exc
                log.warning(f"{LLM} HTTP error: {exc} (attempt {attempt})")
            except Exception as exc:
                last_exc = exc
                log.warning(
                    f"{LLM} Unexpected error: {exc} (attempt {attempt})"
                )

            if attempt < self.max_retries:
                backoff = 2 ** (attempt - 1)
                log.info(f"{LLM} Retrying in {backoff}s …")
                time.sleep(backoff)

        raise ConnectionError(
            f"Ollama API call failed after {self.max_retries} attempts. "
            f"Last error: {last_exc}"
        )
