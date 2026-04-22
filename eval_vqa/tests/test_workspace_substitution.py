"""BUG-046: ${WORKSPACE_DIR} placeholder must be substituted to outputs/ path."""

from __future__ import annotations

from benchmark_runner import (
    S2_LITE,
    S3_ALL,
    S4_ALL,
    _substitute_workspace,
    build_tier_system_prompt,
)


def test_substitute_replaces_dollar_braced_form():
    text = "Write `${WORKSPACE_DIR}/smoke_forward.json` then continue."
    out = _substitute_workspace(text, "/a/b/outputs")
    assert "${WORKSPACE_DIR}" not in out
    assert "/a/b/outputs/smoke_forward.json" in out


def test_substitute_replaces_plain_braced_form():
    text = "Path is {WORKSPACE_DIR}/answer_postprocess.py."
    out = _substitute_workspace(text, "/x/outputs")
    assert "{WORKSPACE_DIR}" not in out
    assert "/x/outputs/answer_postprocess.py" in out


def test_substitute_noop_on_empty():
    assert _substitute_workspace("", "/x/outputs") == ""
    assert _substitute_workspace(None, "/x/outputs") is None  # type: ignore[arg-type]


def test_loaded_skills_have_no_raw_placeholder_after_substitute():
    for name, text in [("S2", S2_LITE), ("S3", S3_ALL), ("S4", S4_ALL)]:
        out = _substitute_workspace(text, "/tmp/outputs")
        assert "${WORKSPACE_DIR}" not in out, f"{name} still has ${'{'}WORKSPACE_DIR{'}'}"
        assert "{WORKSPACE_DIR}" not in out, f"{name} still has {{WORKSPACE_DIR}}"


def test_full_system_prompt_mentions_concrete_outputs():
    prompt = build_tier_system_prompt(
        task_id="pathvqa-task",
        tier="lite",
        question_ids=["pathvqa_00000"],
        subset="all",
        sample_limit=1,
        data_dir="/tmp/public",
        output_dir="/tmp/run/outputs",
    )
    # Concrete artefact paths must appear post-substitution.
    assert "/tmp/run/outputs/smoke_forward.json" in prompt
    assert "/tmp/run/outputs/answer_postprocess.py" in prompt
    # Sandbox rules should explain the placeholder resolution.
    assert "resolve to `/tmp/run/outputs/..." in prompt
