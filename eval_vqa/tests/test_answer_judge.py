"""Unit tests for answer_judge (BUG-038)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from answer_judge import AnswerJudge, _clamp_score, _parse_verdict_json


def _fake_backend(response_map: dict[tuple[str, str], dict[str, Any]]):
    call_count = {"n": 0}

    def _call(system: str, user: str):
        call_count["n"] += 1
        # Key on gold+pred extracted from the user prompt to avoid coupling
        # to exact formatting.
        for (gold, pred), verdict in response_map.items():
            if gold in user and pred in user:
                return verdict, 10, 5
        return {"score": 0, "rationale": "no match"}, 10, 5

    return _call, call_count


def test_clamp_score_rounds_to_triplet():
    assert _clamp_score(1.0) == 1.0
    assert _clamp_score(0.85) == 1.0
    assert _clamp_score(0.5) == 0.5
    assert _clamp_score(0.3) == 0.5
    assert _clamp_score(0.1) == 0.0
    assert _clamp_score("garbage") == 0.0


def test_parse_verdict_json_extracts_embedded_object():
    text = 'Reasoning... {"score": 1, "rationale": "synonym"} trailing'
    parsed = _parse_verdict_json(text)
    assert parsed is not None
    assert parsed["score"] == 1


def test_shortcut_exact_match_skips_backend():
    backend, count = _fake_backend({})
    judge = AnswerJudge(model="test", backend=backend, judge_backend_name="fake")
    verdict = judge.judge_one("q1", "what organ?", "lung", "lung")
    assert verdict.score == 1.0
    assert verdict.judge_backend == "shortcut"
    assert count["n"] == 0


def test_raw_fallback_rescues_truncated_pred():
    # Simulate kimik2.5/slake: raw sentence has gold concept, pred is a
    # 5-word truncation that drops it. Backend returns 0 for pred and 1 for
    # raw; judge_one should return max = 1.
    def backend(system, user):
        if "thoracic cavity" in user.lower() and "chest cavity" in user.lower():
            return {"score": 1, "rationale": "synonym"}, 10, 5
        return {"score": 0, "rationale": "missing"}, 10, 5

    judge = AnswerJudge(model="test", backend=backend, judge_backend_name="fake")
    verdict = judge.judge_one(
        qid="q",
        question="what cavity?",
        gold="thoracic cavity",
        pred="contains the lungs, heart, and",
        raw="contains the lungs, heart, and other structures within the chest cavity",
    )
    assert verdict.score == 1.0
    assert verdict.rationale.startswith("raw-fallback:")


def test_raw_fallback_ignored_when_pred_already_correct():
    calls = {"n": 0}

    def backend(system, user):
        calls["n"] += 1
        return {"score": 1, "rationale": "match"}, 10, 5

    judge = AnswerJudge(model="test", backend=backend, judge_backend_name="fake")
    verdict = judge.judge_one(
        qid="q", question="?", gold="lung", pred="lung tissue",
        raw="lung tissue visible in the image",
    )
    assert verdict.score == 1.0
    # Primary judged; raw skipped since primary already 1.0.
    assert calls["n"] == 1


def test_shortcut_empty_pred_scores_zero():
    backend, count = _fake_backend({})
    judge = AnswerJudge(model="test", backend=backend, judge_backend_name="fake")
    verdict = judge.judge_one("q1", "q", "lung", "")
    assert verdict.score == 0.0
    assert count["n"] == 0


def test_synonym_scored_via_backend(tmp_path: Path):
    backend, count = _fake_backend({
        ("neoplasm", "tumor"): {"score": 1, "rationale": "synonyms"},
    })
    cache = tmp_path / "cache.jsonl"
    judge = AnswerJudge(
        model="test-model",
        backend=backend,
        cache_path=str(cache),
        judge_backend_name="fake",
    )
    verdict = judge.judge_one("q1", "what lesion?", "neoplasm", "tumor")
    assert verdict.score == 1.0
    assert verdict.judge_backend == "fake"
    assert count["n"] == 1
    assert cache.exists()


def test_cache_hit_avoids_second_backend_call(tmp_path: Path):
    backend, count = _fake_backend({
        ("GI", "gastrointestinal"): {"score": 1, "rationale": "expansion"},
    })
    cache = tmp_path / "cache.jsonl"
    judge = AnswerJudge(
        model="test-model",
        backend=backend,
        cache_path=str(cache),
        judge_backend_name="fake",
    )
    v1 = judge.judge_one("q2", "what system?", "GI", "gastrointestinal")
    v2 = judge.judge_one("q2", "what system?", "GI", "gastrointestinal")
    assert v1.score == v2.score == 1.0
    assert v2.cached is True
    assert count["n"] == 1

    # A fresh judge loading the same cache should also hit.
    judge2 = AnswerJudge(
        model="test-model",
        backend=backend,
        cache_path=str(cache),
        judge_backend_name="fake",
    )
    v3 = judge2.judge_one("q2", "what system?", "GI", "gastrointestinal")
    assert v3.cached is True
    assert count["n"] == 1


def test_backend_failure_falls_back_to_heuristic(tmp_path: Path):
    def _broken_backend(system: str, user: str):
        raise RuntimeError("boom")

    judge = AnswerJudge(
        model="test",
        backend=_broken_backend,
        cache_path=str(tmp_path / "c.jsonl"),
        judge_backend_name="fake",
        heuristic_fallback=True,
    )
    verdict = judge.judge_one("q3", "?", "lung", "lung tissue")
    # Heuristic computes F1; "lung" overlap gives some F1 -> 0.5 or higher.
    assert verdict.error and "RuntimeError" in verdict.error
    assert verdict.judge_backend == "heuristic_fallback"
    assert verdict.score in (0.0, 0.5, 1.0)


def test_score_is_in_unit_interval(tmp_path: Path):
    def _weird_backend(system: str, user: str):
        return {"score": "1", "rationale": "stringy"}, 0, 0

    judge = AnswerJudge(
        model="test",
        backend=_weird_backend,
        cache_path=str(tmp_path / "c.jsonl"),
        judge_backend_name="fake",
    )
    v = judge.judge_one("q", "?", "gold", "pred")
    assert 0.0 <= v.score <= 1.0


def test_heuristic_only_when_no_backend(tmp_path: Path):
    judge = AnswerJudge(model="h", backend=None, cache_path=str(tmp_path / "c.jsonl"))
    v = judge.judge_one("q", "?", "tumor present", "tumor")
    # Partial F1 ~ 0.67 -> 0.5 bucket; implementation puts >=0.8 at 1.0.
    assert v.judge_backend == "heuristic"
    assert 0.0 <= v.score <= 1.0


def test_cache_concurrent_writes_yield_valid_jsonl(tmp_path: Path):
    """Parallel sweep workers share the jsonl cache; lines must not interleave."""
    import multiprocessing as mp

    cache_file = tmp_path / "c.jsonl"

    def _worker(idx: int, path: str) -> None:
        j = AnswerJudge(model="m", backend=None, cache_path=path)
        for k in range(20):
            j.judge_one(f"q{idx}-{k}", "?", "gold text here", f"pred-{idx}-{k}")

    procs = [mp.Process(target=_worker, args=(i, str(cache_file))) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"worker crashed: {p.exitcode}"

    with open(cache_file, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert lines, "cache should not be empty"
    for ln in lines:
        json.loads(ln)  # every line must parse — no interleaving
