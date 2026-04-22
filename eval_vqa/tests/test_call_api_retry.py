"""BUG-039: call_api retries transient HTTP errors instead of failing the run."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
import requests

import benchmark_runner


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None,
                 retry_after: str | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers: dict[str, str] = {}
        if retry_after is not None:
            self.headers["Retry-After"] = retry_after

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self) -> dict[str, Any]:
        return self._payload


_OK_PAYLOAD = {
    "choices": [
        {
            "message": {"content": "hi", "tool_calls": []},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


def _call(monkeypatch, responses):
    calls: list[float] = []
    monkeypatch.setattr(benchmark_runner.time, "sleep", lambda s: calls.append(s))
    state = {"i": 0}

    def fake_post(*args, **kwargs):
        r = responses[state["i"]]
        state["i"] += 1
        return r

    monkeypatch.setattr(benchmark_runner.requests, "post", fake_post)
    result = benchmark_runner.call_api(
        api_key="x", model="m", system="s", messages=[], tools=[], reasoning=False,
        base_url="http://example",
    )
    return result, calls


def test_retries_on_429_then_succeeds(monkeypatch):
    responses = [
        _FakeResponse(429, retry_after="0.01"),
        _FakeResponse(200, _OK_PAYLOAD),
    ]
    result, sleeps = _call(monkeypatch, responses)
    assert result["content"] == "hi"
    assert len(sleeps) == 1
    assert sleeps[0] >= 0.01


def test_retries_on_500_with_backoff(monkeypatch):
    responses = [
        _FakeResponse(500),
        _FakeResponse(502),
        _FakeResponse(200, _OK_PAYLOAD),
    ]
    result, sleeps = _call(monkeypatch, responses)
    assert result["content"] == "hi"
    assert len(sleeps) == 2


def test_gives_up_after_4_attempts(monkeypatch):
    responses = [_FakeResponse(503) for _ in range(4)]
    with pytest.raises(requests.HTTPError):
        _call(monkeypatch, responses)


def test_does_not_retry_on_400(monkeypatch):
    responses = [_FakeResponse(400, {})]
    with pytest.raises(requests.HTTPError):
        _call(monkeypatch, responses)


def test_retries_on_connection_error(monkeypatch):
    state = {"i": 0}
    monkeypatch.setattr(benchmark_runner.time, "sleep", lambda s: None)

    def fake_post(*args, **kwargs):
        state["i"] += 1
        if state["i"] < 3:
            raise requests.ConnectionError("boom")
        return _FakeResponse(200, _OK_PAYLOAD)

    monkeypatch.setattr(benchmark_runner.requests, "post", fake_post)
    result = benchmark_runner.call_api(
        api_key="x", model="m", system="s", messages=[], tools=[], reasoning=False,
        base_url="http://example",
    )
    assert result["content"] == "hi"
    assert state["i"] == 3
