# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Generic asynchronous client for Jev-compatible System One decisions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from time import monotonic
from typing import Any

import httpx

from openviking.telemetry import tracer
from openviking_cli.utils.config.jev_config import JevConfig
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class JevError(RuntimeError):
    """Raised when a System One request or response is invalid."""


class JevPayloadTooLarge(JevError):
    """Raised when one indivisible Jev request exceeds the configured/provider budget."""


def estimate_jev_input_tokens(
    *,
    state: Any,
    questions: dict[str, dict[str, Any]],
    model: str | None = None,
) -> int:
    """Conservatively estimate tokens without coupling to a provider tokenizer."""
    payload: dict[str, Any] = {"state": state, "questions": questions}
    if model:
        payload["model"] = model
    serialized = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    non_ascii = sum(ord(char) > 127 for char in serialized)
    ascii_chars = len(serialized) - non_ascii
    return max(1, non_ascii + (ascii_chars + 3) // 4)


@dataclass(slots=True)
class JevClient:
    """Evaluate typed questions against shared state through System One."""

    config: JevConfig
    client: Any = None

    @tracer("model.jev.evaluate", ignore_args=True, ignore_result=True)
    async def evaluate(
        self,
        *,
        state: Any,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not questions:
            return {}
        tracer.set("jev.question_count", len(questions))
        tracer.set("jev.state_chars", len(json.dumps(state, ensure_ascii=False, default=str)))
        tracer.set(
            "jev.estimated_input_tokens",
            estimate_jev_input_tokens(
                state=state,
                questions=questions,
                model=self.config.model,
            ),
        )

        owns_client = self.client is None
        headers = httpx.Headers(self.config.extra_headers)
        headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers["Content-Type"] = "application/json"
        client = self.client or httpx.AsyncClient(
            headers=headers,
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
        )
        stats = {"batch_count": 0, "split_count": 0}
        try:
            answers = await self._evaluate_with_client(
                client,
                state=state,
                questions=questions,
                stats=stats,
            )
        finally:
            if owns_client:
                await client.aclose()
        tracer.set("jev.batch_count", stats["batch_count"])
        tracer.set("jev.split_count", stats["split_count"])
        return answers

    async def _evaluate_with_client(
        self,
        client: Any,
        *,
        state: Any,
        questions: dict[str, dict[str, Any]],
        stats: dict[str, int],
    ) -> dict[str, dict[str, Any]]:
        estimated_tokens = estimate_jev_input_tokens(
            state=state,
            questions=questions,
            model=self.config.model,
        )
        if estimated_tokens > self.config.max_input_tokens:
            return await self._split_or_reject(
                client,
                state=state,
                questions=questions,
                stats=stats,
                reason=(
                    f"estimated input {estimated_tokens} exceeds configured Jev budget "
                    f"{self.config.max_input_tokens}"
                ),
            )

        payload: dict[str, Any] = {"state": state, "questions": questions}
        if self.config.model:
            payload["model"] = self.config.model
        stats["batch_count"] += 1
        try:
            body = await self._post_with_retry(client, payload)
        except JevPayloadTooLarge as exc:
            return await self._split_or_reject(
                client,
                state=state,
                questions=questions,
                stats=stats,
                reason=str(exc),
            )
        answers = body.get("answers") if isinstance(body, dict) else None
        if not isinstance(answers, dict):
            raise JevError("Jev decision response has no answers object")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        for key in ("input_tokens", "state_tokens", "question_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                tracer.set(f"jev.{key}", value)
        return answers

    async def _split_or_reject(
        self,
        client: Any,
        *,
        state: Any,
        questions: dict[str, dict[str, Any]],
        stats: dict[str, int],
        reason: str,
    ) -> dict[str, dict[str, Any]]:
        if len(questions) <= 1:
            question_id = next(iter(questions), "<none>")
            raise JevPayloadTooLarge(f"Jev question {question_id} is oversized: {reason}")
        stats["split_count"] += 1
        items = list(questions.items())
        midpoint = len(items) // 2
        answers: dict[str, dict[str, Any]] = {}
        for part in (items[:midpoint], items[midpoint:]):
            answers.update(
                await self._evaluate_with_client(
                    client,
                    state=state,
                    questions=dict(part),
                    stats=stats,
                )
            )
        return answers

    async def _post_with_retry(self, client: Any, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.config.max_retries + 1):
            started = monotonic()
            response = None
            try:
                response = await client.post(self.config.api_url, json=payload)
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise JevError("Jev decision response must be a JSON object")
                return body
            except Exception as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    response = exc.response
                oversized = isinstance(exc, JevPayloadTooLarge) or _payload_too_large_jev_error(exc)
                retrying = (
                    not oversized
                    and attempt < self.config.max_retries
                    and _retryable_jev_error(exc)
                )
                delay = _retry_delay(self.config, attempt=attempt, error=exc) if retrying else 0
                headers = getattr(response, "headers", {})
                # Do not log the payload, response body, URL or exception message: they may
                # contain session content or credentials. No response means no HTTP code.
                logger.warning(
                    "[JEV_REQUEST_FAILED] %s",
                    json.dumps(
                        {
                            "trace_id": tracer.get_trace_id(),
                            "attempt": attempt + 1,
                            "max_attempts": self.config.max_retries + 1,
                            "elapsed_s": round(monotonic() - started, 3),
                            "timeout_s": self.config.timeout,
                            "error_type": type(exc).__name__,
                            "http_status": getattr(response, "status_code", None),
                            "provider_error_code": _response_error_code(response),
                            "server_request_id": headers.get("x-request-id")
                            or headers.get("x-tt-logid"),
                            "retrying": retrying,
                            "retry_delay_s": delay,
                        },
                        ensure_ascii=False,
                    ),
                )
                if isinstance(exc, JevPayloadTooLarge):
                    raise
                if oversized:
                    raise JevPayloadTooLarge(
                        f"Jev provider rejected oversized input with HTTP "
                        f"{exc.response.status_code}"
                    ) from exc
                if not retrying:
                    raise JevError(f"Jev decision request failed: {exc}") from exc
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")


def _response_error_code(response: Any) -> str | int | None:
    if response is None:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error", body) if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str):
        return code[:128]
    return code if type(code) is int else None


def _retryable_jev_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504, 529}
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _payload_too_large_jev_error(exc: Exception) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    if exc.response.status_code == 413:
        return True
    if exc.response.status_code != 422:
        return False
    body = exc.response.text.casefold()
    return any(
        marker in body
        for marker in (
            "maximum context length",
            "context length",
            "input_tokens",
            "too large",
            "too long",
        )
    )


def _retry_delay(config: JevConfig, *, attempt: int, error: Exception) -> float:
    delay = config.retry_backoff_seconds * (2**attempt)
    if isinstance(error, httpx.HTTPStatusError):
        retry_after = error.response.headers.get("Retry-After")
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
    return min(delay, config.timeout)
