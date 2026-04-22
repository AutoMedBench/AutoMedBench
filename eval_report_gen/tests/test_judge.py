#!/usr/bin/env python3
from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from eval_report_gen.llm_judge import (
    HeuristicJudge,
    _encode_png_data_url,
    _extract_json,
    build_online_judge_user_content,
)


MINIMAL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+aG7QAAAAASUVORK5CYII="
)


class JudgeTests(unittest.TestCase):
    def test_extract_json_from_wrapped_text(self):
        payload = _extract_json("prefix {\"s1_plan_score\": 1.0} suffix")
        self.assertEqual(payload["s1_plan_score"], 1.0)

    def test_heuristic_judge_returns_contract(self):
        judge = HeuristicJudge()
        verdict = judge.judge(
            conversation={
                "tier": "lite",
                "messages": [{"role": "user", "content": "validate one case first"}],
                "code_executions": [{"exit_code": 0, "code": "print('validate')"}],
                "artifact_paths": {},
            },
            eval_report={"metrics": {}, "format": {}, "aggregate": {}},
            task="mimic-cxr-report-task",
        )
        payload = verdict.to_dict()
        self.assertIn("s1_plan_score", payload)
        self.assertIn("s2_setup_score", payload)
        self.assertIn("s3_validate_score", payload)

    def test_encode_png_data_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "plan.png"
            png_path.write_bytes(base64.b64decode(MINIMAL_PNG_BASE64))
            data_url = _encode_png_data_url(str(png_path))
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

    def test_online_user_content_includes_plan_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            plan_dir = output_dir / "plan"
            agent_outputs = output_dir / "agent_outputs"
            plan_dir.mkdir(parents=True)
            agent_outputs.mkdir(parents=True)
            (plan_dir / "plan.md").write_text("# plan\n", encoding="utf-8")
            (plan_dir / "plan.png").write_bytes(base64.b64decode(MINIMAL_PNG_BASE64))
            user_content = build_online_judge_user_content(
                conversation={
                    "tier": "standard",
                    "messages": [],
                    "code_executions": [],
                    "artifact_paths": {
                        "output_dir": str(output_dir),
                        "plan_md": str(plan_dir / "plan.md"),
                        "plan_png": str(plan_dir / "plan.png"),
                        "agent_outputs": str(agent_outputs),
                    },
                },
                eval_report={"metrics": {}, "format": {}, "aggregate": {}},
                task="mimic-cxr-report-task",
            )
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
