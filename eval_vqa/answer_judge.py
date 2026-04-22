#!/usr/bin/env python3
"""Answer-level LLM judge for open-ended VQA (BUG-038).

Token F1 penalises legitimate synonyms (tumor vs neoplasm, GI vs
gastrointestinal, etc.) in open-ended medical VQA. This module wraps a cheap
LLM (Claude Haiku 4.5 via OpenRouter-compatible endpoint by default) in a
judge that returns a 0 / 0.5 / 1 score per (question, gold, pred) triple,
with a persistent jsonl cache and a heuristic fallback when no key is set.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
try:
    import fcntl  # POSIX only; used for cross-process jsonl cache locking.
except ImportError:  # pragma: no cover - non-POSIX fallback.
    fcntl = None  # type: ignore[assignment]
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from answer_metrics import exact_match, token_f1, yes_no_accuracy
from answer_normalizer import is_yes_no_answer


JUDGE_SYSTEM_PROMPT = (
    "You are a medical VQA grader. Given a question, the gold answer, and a "
    "model prediction, decide if the prediction is clinically correct.\n"
    "Score rules:\n"
    "  1.0 — clinically equivalent (synonyms, paraphrase, acronym/expansion, "
    "extra hedging that does not change meaning).\n"
    "  0.5 — partially correct (covers some of the gold concept but omits or "
    "confuses a meaningful component).\n"
    "  0.0 — incorrect, unrelated, contradictory, or pure speculation.\n"
    "Reply with ONLY a single JSON object: "
    '{"score": 0|0.5|1, "rationale": "<=20 words"}. '
    "Do not add any other text."
)


@dataclass
class AnswerVerdict:
    qid: str
    score: float = 0.0
    rationale: str = ""
    judge_model: str = ""
    judge_backend: str = "heuristic"
    latency_s: float = 0.0
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "score": self.score,
            "rationale": self.rationale,
            "judge_model": self.judge_model,
            "judge_backend": self.judge_backend,
            "latency_s": self.latency_s,
            "cached": self.cached,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }


BackendFn = Callable[[str, str], tuple[dict[str, Any], int, int]]
# Signature: (system_prompt, user_prompt) -> (parsed_json, input_tokens, output_tokens)


class _JsonlCache:
    """Append-only jsonl cache keyed by sha256(qid||gold||pred)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._store: dict[str, dict[str, Any]] = {}
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        key = entry.get("key")
                        if key:
                            self._store[key] = entry

            except OSError:
                pass

    @staticmethod
    def make_key(qid: str, gold: str, pred: str, model: str) -> str:
        payload = f"{qid}\x1f{gold}\x1f{pred}\x1f{model}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self._store.get(key)

    def put(self, key: str, verdict_dict: dict[str, Any]) -> None:
        entry = {"key": key, **verdict_dict}
        self._store[key] = entry
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        # Concurrent sweep workers may share this file — lock to prevent
        # interleaved JSONL lines and double-write of the same key.
        line = json.dumps(entry) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                except OSError:
                    pass
            try:
                handle.write(line)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass


def _heuristic_score(question: str, gold: str, pred: str) -> tuple[float, str]:
    gold = (gold or "").strip()
    pred = (pred or "").strip()
    if not gold or not pred:
        return 0.0, "empty input"
    if is_yes_no_answer(gold):
        yn = yes_no_accuracy(pred, gold) / 100.0
        return (1.0 if yn >= 1.0 else 0.0), "yes/no heuristic"
    em = exact_match(pred, gold) / 100.0
    f1 = token_f1(pred, gold) / 100.0
    if em >= 1.0 or f1 >= 0.8:
        return 1.0, "heuristic: high F1/EM"
    if f1 >= 0.4:
        return 0.5, "heuristic: partial F1"
    return 0.0, "heuristic: low F1"


def _parse_verdict_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    # Try direct parse first, then extract first JSON object.
    candidates: list[str] = [text]
    match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "score" in data:
            return data
    return None


def _clamp_score(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value >= 0.75:
        return 1.0
    if value >= 0.25:
        return 0.5
    return 0.0


def _default_openai_backend(
    api_key: str,
    base_url: str,
    model: str,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> BackendFn:
    endpoint = base_url.rstrip("/") + "/chat/completions"

    # NVDA-hosted Claude reasoning models (bedrock-claude-*) reject
    # `temperature`; OpenAI o-series does the same. Omit the field for them.
    _reasoning = (
        "bedrock-claude" in model
        or model.startswith(("o1-", "o3-", "o4-"))
    )

    def _call(system: str, user: str) -> tuple[dict[str, Any], int, int]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 200,
        }
        if not _reasoning:
            payload["temperature"] = 0.0
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps(payload)
        transient = {429, 500, 502, 503, 504}
        last_exc: Exception | None = None
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(endpoint, headers=headers, data=body, timeout=timeout)
                if response.status_code in transient and attempt < max_retries:
                    wait = min(30.0, 2 ** attempt + random.uniform(0, 1))
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt == max_retries:
                    raise
                time.sleep(min(30.0, 2 ** attempt + random.uniform(0, 1)))
        if response is None:
            if last_exc:
                raise last_exc
            raise RuntimeError("judge backend: no response")
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"judge API error: {data['error']}")
        content = data["choices"][0]["message"].get("content") or ""
        parsed = _parse_verdict_json(content) or {}
        usage = data.get("usage", {})
        return parsed, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    return _call


class AnswerJudge:
    """LLM-as-judge for open-ended VQA answers."""

    def __init__(
        self,
        model: str = "anthropic/claude-haiku-4.5",
        backend: BackendFn | None = None,
        cache_path: str | None = None,
        judge_backend_name: str | None = None,
        heuristic_fallback: bool = True,
    ) -> None:
        self.model = model
        self.backend = backend
        self.cache = _JsonlCache(cache_path or "")
        self.judge_backend_name = judge_backend_name or (
            "heuristic" if backend is None else "llm"
        )
        self.heuristic_fallback = heuristic_fallback

    @classmethod
    def from_env(
        cls,
        cache_path: str | None = None,
        model: str | None = None,
    ) -> "AnswerJudge":
        # Pair each provider key with its matching base_url so we never send
        # a key to the wrong endpoint (prior bug: NVDA key sent to OpenRouter
        # → 401). Explicit ANSWER_JUDGE_BASE_URL still wins.
        env_base_url = os.environ.get("ANSWER_JUDGE_BASE_URL")
        key_providers = [
            ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
            ("OPENAI_API_KEY", "https://api.openai.com/v1"),
            ("NVDA_API_KEY", "https://inference-api.nvidia.com"),
        ]
        api_key: str | None = None
        base_url = env_base_url or ""
        for env_name, default_url in key_providers:
            val = os.environ.get(env_name)
            if val:
                api_key = val
                if not env_base_url:
                    base_url = default_url
                break
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"

        chosen_model = model or os.environ.get(
            "ANSWER_JUDGE_MODEL", "anthropic/claude-haiku-4.5"
        )
        if not api_key:
            return cls(model=chosen_model, backend=None, cache_path=cache_path)
        backend = _default_openai_backend(api_key, base_url, chosen_model)
        backend_name = (
            "nvda" if "nvidia.com" in base_url
            else "openai" if "openai.com" in base_url
            else "openrouter"
        )
        # Log once so failures are easy to trace (key/url mismatch is a
        # recurring 401 source).
        import sys as _sys
        _sys.stderr.write(
            f"[answer_judge] backend={backend_name} model={chosen_model} "
            f"base_url={base_url}\n"
        )
        _sys.stderr.flush()
        return cls(
            model=chosen_model,
            backend=backend,
            cache_path=cache_path,
            judge_backend_name=backend_name,
        )

    def judge_one(
        self,
        qid: str,
        question: str,
        gold: str,
        pred: str,
        raw: str = "",
    ) -> AnswerVerdict:
        """Judge pred against gold.

        If ``raw`` is provided and differs from ``pred``, judge it separately
        and return whichever verdict scores higher. This rescues runs where a
        weak agent-authored ``answer_postprocess.py`` truncated a correct
        full-sentence model output into a wrong short span (e.g. kimik2.5
        slake F: raw="contains the lungs, heart, and other structures within
        the chest cavity" but pred="contains the lungs, heart, and").
        """
        primary = self._judge_single(qid, question, gold, pred)
        raw_text = (raw or "").strip()
        pred_text = (pred or "").strip()
        if (
            raw_text
            and raw_text.lower() != pred_text.lower()
            and len(raw_text) > len(pred_text)
            and primary.score < 1.0
        ):
            secondary = self._judge_single(qid, question, gold, raw_text)
            if secondary.score > primary.score:
                secondary.rationale = (
                    f"raw-fallback: {secondary.rationale}"[:200]
                )
                return secondary
        return primary

    def _judge_single(self, qid: str, question: str, gold: str, pred: str) -> AnswerVerdict:
        gold = (gold or "").strip()
        pred = (pred or "").strip()
        # Fast paths: empty pred -> 0; exact match -> 1.
        if not pred:
            return AnswerVerdict(
                qid=qid, score=0.0, rationale="empty prediction",
                judge_model=self.model, judge_backend="shortcut",
            )
        if gold and pred.lower() == gold.lower():
            return AnswerVerdict(
                qid=qid, score=1.0, rationale="exact string match",
                judge_model=self.model, judge_backend="shortcut",
            )
        key = self.cache.make_key(qid, gold, pred, self.model)
        cached = self.cache.get(key)
        if cached is not None:
            return AnswerVerdict(
                qid=qid,
                score=float(cached.get("score", 0.0)),
                rationale=str(cached.get("rationale", "")),
                judge_model=str(cached.get("judge_model", self.model)),
                judge_backend=str(cached.get("judge_backend", self.judge_backend_name)),
                latency_s=float(cached.get("latency_s", 0.0)),
                cached=True,
                input_tokens=int(cached.get("input_tokens", 0)),
                output_tokens=int(cached.get("output_tokens", 0)),
                error=cached.get("error"),
            )
        if self.backend is None:
            score, rationale = _heuristic_score(question, gold, pred)
            verdict = AnswerVerdict(
                qid=qid, score=score, rationale=rationale,
                judge_model=self.model, judge_backend="heuristic",
            )
            self.cache.put(key, verdict.to_dict())
            return verdict
        user_prompt = (
            f"Question: {question or '(not provided)'}\n"
            f"Gold answer: {gold}\n"
            f"Model prediction: {pred}\n"
            "Output the JSON verdict now."
        )
        started = time.time()
        try:
            parsed, in_tok, out_tok = self.backend(JUDGE_SYSTEM_PROMPT, user_prompt)
            score = _clamp_score(parsed.get("score"))
            rationale = str(parsed.get("rationale") or "").strip()[:200]
            verdict = AnswerVerdict(
                qid=qid,
                score=score,
                rationale=rationale or "llm verdict",
                judge_model=self.model,
                judge_backend=self.judge_backend_name,
                latency_s=round(time.time() - started, 4),
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        except Exception as exc:  # noqa: BLE001 - judge must not crash eval
            sys.stderr.write(f"[answer_judge] backend failure: {type(exc).__name__}: {exc}\n")
            if self.heuristic_fallback:
                score, rationale = _heuristic_score(question, gold, pred)
                verdict = AnswerVerdict(
                    qid=qid, score=score, rationale=f"fallback: {rationale}",
                    judge_model=self.model, judge_backend="heuristic_fallback",
                    latency_s=round(time.time() - started, 4),
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                verdict = AnswerVerdict(
                    qid=qid, score=0.0, rationale="backend error",
                    judge_model=self.model, judge_backend=self.judge_backend_name,
                    latency_s=round(time.time() - started, 4),
                    error=f"{type(exc).__name__}: {exc}",
                )
        self.cache.put(key, verdict.to_dict())
        return verdict


def judge_agreement_rate(per_question: dict[str, dict[str, Any]]) -> float:
    """Fraction of samples where judge (>=0.5) agrees with token F1 >= 0.5.

    Returns 0.0 when no open-ended samples with judge scores are present.
    """
    total = 0
    agree = 0
    for row in per_question.values():
        if "judge_score" not in row:
            continue
        total += 1
        judge_pos = float(row.get("judge_score", 0.0)) >= 0.5
        # Per BUG-037, sample_score for open-ended is 0.5*EM + 0.5*F1 (or yes/no).
        # We approximate F1>=0.5 via the stored sample_score when available;
        # fall back to 'correct' bool.
        sample_score = float(row.get("sample_score", 0.0))
        heuristic_pos = sample_score >= 0.5 or bool(row.get("correct"))
        if judge_pos == heuristic_pos:
            agree += 1
    return round(agree / total, 4) if total > 0 else 0.0
