"""Unit tests for BUG-1 / BUG-4 / BUG-5 guards."""

from __future__ import annotations

import json
import os

import pytest

from aggregate import (
    compute_s2_binary,
    compute_s4_with_guards,
    infer_inference_mode,
)
from format_checker import check_answer_file, check_submission, detect_placeholder
from inference_verifier import (
    check_postprocess_artefact,
    check_smoke_forward,
    detect_model_call,
)


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _answer(raw: str, answer: str = "yes", label: str = "A") -> dict:
    return {
        "question_id": "q_00000",
        "predicted_label": label,
        "predicted_answer": answer,
        "raw_model_output": raw,
        "model_name": "test-model",
        "runtime_s": 1.0,
    }


# -------- BUG-1: detect_placeholder --------

def test_detect_placeholder_empty():
    # Honest-skip path: both raw and answer empty → not a placeholder,
    # scorer simply drops the sample from valid_outputs.
    assert detect_placeholder("", "")[0] is False
    assert detect_placeholder("   ", "")[0] is False
    assert detect_placeholder(None, None)[0] is False
    # Half-empty (claimed answer but no model text) is still flagged.
    assert detect_placeholder("", "yes")[0] is True
    assert detect_placeholder(None, "yes")[0] is True


def test_detect_placeholder_whole_words():
    for val in ("unknown", "UNKNOWN", "n/a", "none"):
        assert detect_placeholder(val)[0] is True, val


def test_detect_placeholder_prefixes():
    for val in (
        "heuristic:papillary thyroid carcinoma",
        "fallback:pathology finding",
        "PLACEHOLDER: something",
        "dummy: x",
    ):
        assert detect_placeholder(val)[0] is True, val


def test_detect_placeholder_real_text():
    real = "The image shows a histopathological view of tissue with branching papillae."
    assert detect_placeholder(real)[0] is False


def test_check_answer_file_flags_placeholder(tmp_path):
    question = {"question_id": "q_00000", "options": {}}
    ans_path = tmp_path / "q_00000" / "answer.json"
    _write(str(ans_path), json.dumps(_answer(raw="heuristic:foo")))
    res = check_answer_file(str(ans_path), question)
    assert res["is_placeholder"] is True
    assert res["parsed"] is False


def test_check_answer_file_accepts_real(tmp_path):
    question = {"question_id": "q_00000", "options": {}}
    ans_path = tmp_path / "q_00000" / "answer.json"
    _write(str(ans_path), json.dumps(_answer(raw="Real decoded output from LLaVA-Med about the image.")))
    res = check_answer_file(str(ans_path), question)
    assert res["is_placeholder"] is False
    assert res["parsed"] is True


def test_check_submission_placeholder_rate(tmp_path):
    public = tmp_path / "public"
    pred = tmp_path / "pred"
    qids = []
    for i, raw in enumerate(["heuristic:x", "", "real decoded text about tissue morphology", "fallback:y"]):
        qid = f"q_{i:03d}"
        qids.append(qid)
        _write(str(public / qid / "question.json"), json.dumps({"question_id": qid, "options": {}}))
        _write(str(pred / qid / "answer.json"), json.dumps(_answer(raw=raw) | {"question_id": qid}))
    report = check_submission(agent_dir=str(pred), question_ids=qids, public_dir=str(public))
    assert report["counts"]["placeholder_files"] == 3
    assert abs(report["placeholder_rate"] - 0.75) < 1e-6


# -------- BUG-4: detect_model_call --------

def _make_conv(tmp_path, code_executions):
    path = tmp_path / "conversation.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"code_executions": code_executions, "messages": []}, fh)
    return str(path)


def test_detect_model_call_positive(tmp_path):
    conv = _make_conv(
        tmp_path,
        [
            {
                "turn": 1,
                "language": "python",
                "code": "from transformers import AutoModelForCausalLM\n"
                        "model = AutoModelForCausalLM.from_pretrained('x')\n"
                        "out = model.generate(inputs)",
                "exit_code": 0,
                "stdout_preview": "Loading checkpoint shards: 100%",
                "exec_time_s": 5.0,
            }
        ],
    )
    info = detect_model_call(conv)
    assert info["detected"] is True
    assert info["has_generate"] is True
    assert info["has_loader"] is True


def test_detect_model_call_no_generate(tmp_path):
    conv = _make_conv(
        tmp_path,
        [
            {
                "turn": 1,
                "language": "bash",
                "code": "echo 'just setup'",
                "exit_code": 0,
                "stdout_preview": "ok",
                "exec_time_s": 0.1,
            }
        ],
    )
    info = detect_model_call(conv)
    assert info["detected"] is False


def test_detect_model_call_shortcuts(tmp_path):
    conv = _make_conv(
        tmp_path,
        [
            {
                "turn": 4,
                "language": "python",
                "code": "def h(q): return 'heuristic:' + q\n",
                "exit_code": 0,
                "stdout_preview": "",
                "exec_time_s": 0.01,
            }
        ],
    )
    info = detect_model_call(conv)
    assert info["detected"] is False
    assert any("heuristic" in s for s in info["shortcuts"])


def test_detect_model_call_failed_load_not_detected(tmp_path):
    # from_pretrained attempted but failed — exit_code != 0 so not counted.
    conv = _make_conv(
        tmp_path,
        [
            {
                "turn": 2,
                "language": "python",
                "code": "m = AutoModelForCausalLM.from_pretrained('x'); m.generate()",
                "exit_code": 1,
                "stdout_preview": "OSError: no such file",
                "exec_time_s": 0.5,
            }
        ],
    )
    info = detect_model_call(conv)
    assert info["detected"] is False


# -------- BUG-5: check_smoke_forward --------

def test_smoke_forward_missing(tmp_path):
    res = check_smoke_forward(str(tmp_path))
    assert res["valid"] is False
    assert res["reason"] == "missing"


def test_smoke_forward_valid(tmp_path):
    record = {
        "model_name": "llava-med",
        "device": "cuda:0",
        "wall_s": 1.5,
        "raw_output_sample": "The image shows histopathological features of ...",
        "success": True,
    }
    _write(str(tmp_path / "smoke_forward.json"), json.dumps(record))
    res = check_smoke_forward(str(tmp_path))
    assert res["valid"] is True


def test_smoke_forward_too_fast(tmp_path):
    record = {
        "model_name": "llava-med",
        "device": "cuda:0",
        "wall_s": 0.01,
        "raw_output_sample": "ok",
        "success": True,
    }
    _write(str(tmp_path / "smoke_forward.json"), json.dumps(record))
    res = check_smoke_forward(str(tmp_path))
    assert res["valid"] is False
    assert "wall_s" in res["reason"]


def test_smoke_forward_success_false(tmp_path):
    record = {
        "model_name": "llava-med",
        "device": "cuda:0",
        "wall_s": 1.0,
        "raw_output_sample": "The image shows histo features",
        "success": False,
    }
    _write(str(tmp_path / "smoke_forward.json"), json.dumps(record))
    res = check_smoke_forward(str(tmp_path))
    assert res["valid"] is False
    assert res["reason"] == "success=False"


def test_smoke_forward_placeholder_sample(tmp_path):
    record = {
        "model_name": "llava-med",
        "device": "cuda:0",
        "wall_s": 1.0,
        "raw_output_sample": "heuristic:yes",
        "success": True,
    }
    _write(str(tmp_path / "smoke_forward.json"), json.dumps(record))
    res = check_smoke_forward(str(tmp_path))
    assert res["valid"] is False
    assert "placeholder" in res["reason"]


# -------- P1-A: postprocess artefact --------

_PP_GOOD = """\
def postprocess(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("yes"):
        return "yes"
    if raw.startswith("no"):
        return "no"
    return raw.split('.')[0]
"""


def _write_cal(path: str, n: int) -> None:
    recs = [
        {
            "question_id": f"q_{i:03d}",
            "raw_model_output": "No, tissue normal.",
            "predicted_answer": "no",
            "gold_answer": "no",
            "hit": True,
        }
        for i in range(n)
    ]
    _write(path, json.dumps(recs))


def test_postprocess_missing(tmp_path):
    res = check_postprocess_artefact(str(tmp_path))
    assert res["valid"] is False
    assert "missing" in res["reason"]


def test_postprocess_valid(tmp_path):
    _write(str(tmp_path / "answer_postprocess.py"), _PP_GOOD)
    _write_cal(str(tmp_path / "s3_calibration.json"), 15)
    res = check_postprocess_artefact(str(tmp_path))
    assert res["valid"] is True, res
    assert res["postprocess_ok"] is True
    assert res["calibration_ok"] is True
    assert res["calibration_n"] == 15


def test_postprocess_too_few_samples(tmp_path):
    _write(str(tmp_path / "answer_postprocess.py"), _PP_GOOD)
    _write_cal(str(tmp_path / "s3_calibration.json"), 5)
    res = check_postprocess_artefact(str(tmp_path))
    assert res["valid"] is False
    assert "10" in res["reason"]


def test_postprocess_no_callable(tmp_path):
    _write(str(tmp_path / "answer_postprocess.py"), "not_postprocess = lambda x: x\n")
    _write_cal(str(tmp_path / "s3_calibration.json"), 15)
    res = check_postprocess_artefact(str(tmp_path))
    assert res["valid"] is False
    assert "callable" in res["reason"] or "postprocess" in res["reason"].lower()


def test_postprocess_raises(tmp_path):
    _write(
        str(tmp_path / "answer_postprocess.py"),
        "def postprocess(raw):\n    raise RuntimeError('boom')\n",
    )
    _write_cal(str(tmp_path / "s3_calibration.json"), 15)
    res = check_postprocess_artefact(str(tmp_path))
    assert res["valid"] is False
    assert "raise" in res["reason"].lower() or "boom" in res["reason"]


# -------- BUG-042: S3 calibration anti-cheat --------

def _write_cal_custom(path: str, recs: list[dict]) -> None:
    _write(path, json.dumps(recs))


def test_calibration_gold_unknown_is_rejected(tmp_path):
    """BUG-042: gold_answer='unknown' across most records → fake hits, reject."""
    _write(str(tmp_path / "answer_postprocess.py"), _PP_GOOD)
    recs = [
        {
            "question_id": f"q_{i:03d}",
            "raw_model_output": "No, tissue appears normal in this slide.",
            "predicted_answer": "no",
            "gold_answer": "unknown",
            "hit": True,
        }
        for i in range(15)
    ]
    _write_cal_custom(str(tmp_path / "s3_calibration.json"), recs)
    res = check_postprocess_artefact(str(tmp_path))
    assert res["valid"] is False, res
    assert res["invalid_gold_rate"] == 1.0
    assert "unknown" in res["reason"].lower() or "fake" in res["reason"].lower()


def test_calibration_malformed_raw_is_rejected(tmp_path):
    """BUG-042: raw_model_output truncated to '-glass' / 'oma.' → reject."""
    _write(str(tmp_path / "answer_postprocess.py"), _PP_GOOD)
    truncated = ["-glass nuclei.", "oma.", ", which is benign.", "ically removed."]
    recs = [
        {
            "question_id": f"q_{i:03d}",
            "raw_model_output": truncated[i % len(truncated)],
            "predicted_answer": "x",
            "gold_answer": "lung",
            "hit": False,
        }
        for i in range(15)
    ]
    _write_cal_custom(str(tmp_path / "s3_calibration.json"), recs)
    res = check_postprocess_artefact(str(tmp_path))
    assert res["valid"] is False, res
    assert res["malformed_raw_rate"] > 0.2
    assert "truncated" in res["reason"].lower() or "decode" in res["reason"].lower()


def test_calibration_valid_with_real_gold_passes(tmp_path):
    """BUG-042: real gold + well-formed raw should still pass."""
    _write(str(tmp_path / "answer_postprocess.py"), _PP_GOOD)
    recs = [
        {
            "question_id": f"q_{i:03d}",
            "raw_model_output": "No, the tissue shows mild edema only.",
            "predicted_answer": "no",
            "gold_answer": "no" if i % 2 == 0 else "yes",
            "hit": i % 2 == 0,
        }
        for i in range(15)
    ]
    _write_cal_custom(str(tmp_path / "s3_calibration.json"), recs)
    res = check_postprocess_artefact(str(tmp_path))
    assert res["valid"] is True, res
    assert res["invalid_gold_rate"] == 0.0
    assert res["malformed_raw_rate"] == 0.0
    # Effective hit rate = 8/15 = 0.5333 (rounded to 4 decimals)
    assert abs(res["calibration_hit_rate"] - 0.5333) < 0.01
    assert res["calibration_hit_rate_raw"] == res["calibration_hit_rate"]


def test_calibration_mixed_tolerates_below_20pct(tmp_path):
    """BUG-042: 2/15 invalid gold (13.3%) should pass; 4/15 (26.6%) should fail."""
    _write(str(tmp_path / "answer_postprocess.py"), _PP_GOOD)
    # 2 unknown, 13 real gold → 13.3%, passes threshold
    recs = [
        {
            "question_id": f"q_{i:03d}",
            "raw_model_output": "No, the tissue shows no abnormalities.",
            "predicted_answer": "no",
            "gold_answer": "unknown" if i < 2 else "no",
            "hit": i >= 2,
        }
        for i in range(15)
    ]
    _write_cal_custom(str(tmp_path / "s3_calibration.json"), recs)
    res = check_postprocess_artefact(str(tmp_path))
    assert res["valid"] is True, res
    # Effective hit rate excludes the 2 unknown-gold records: 13/13 = 1.0
    assert res["calibration_hit_rate"] == 1.0
    assert res["invalid_gold_n"] == 2


# -------- P1-B: quality floor + real_but_broken --------

def test_s4_quality_floor_real_but_broken():
    s, pens = compute_s4_with_guards(
        completion_rate=1.0,
        parse_rate=1.0,
        placeholder_rate=0.0,
        model_call_detected=True,
        accuracy=0.001,
    )
    assert s == 0.5, s
    assert any("real_but_broken" in p for p in pens)


def test_s4_healthy_real_not_capped():
    s, pens = compute_s4_with_guards(
        completion_rate=1.0,
        parse_rate=1.0,
        placeholder_rate=0.0,
        model_call_detected=True,
        accuracy=0.31,
    )
    assert s == 1.0
    assert pens == []


def test_infer_mode_real_but_broken():
    m = infer_inference_mode(
        placeholder_rate=0.0,
        placeholder_reasons={},
        completion_rate=1.0,
        valid_output_rate=1.0,
        accuracy=0.001,
        model_call_detected=True,
    )
    assert m == "real_but_broken"


def test_infer_mode_real_healthy():
    m = infer_inference_mode(
        placeholder_rate=0.0,
        placeholder_reasons={},
        completion_rate=1.0,
        valid_output_rate=1.0,
        accuracy=0.30,
        model_call_detected=True,
    )
    assert m == "real"


# -------- P3: S2 binary rubric --------

def test_s2_binary_all_pass():
    s, comps = compute_s2_binary(
        smoke_forward_passed=True, model_call_detected=True, env_ready=True
    )
    assert s == 1.0
    assert all(comps.values())


def test_s2_binary_smoke_missing():
    s, comps = compute_s2_binary(
        smoke_forward_passed=False, model_call_detected=True, env_ready=True
    )
    assert abs(s - 2 / 3) < 1e-3
    assert comps["smoke_forward_passed"] is False


def test_s2_binary_all_fail():
    s, comps = compute_s2_binary(
        smoke_forward_passed=False, model_call_detected=False, env_ready=False
    )
    assert s == 0.0
