#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

from eval_report_gen.run_eval import run_eval
from eval_report_gen.task_loader import discover_cases, get_task_data_root, load_task_config


class EvalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_task_config("mimic-cxr-report-task")
        self.cases = discover_cases("mimic-cxr-report-task")
        self.gt_dir = str(Path(get_task_data_root("mimic-cxr-report-task")) / "private")

    def _run(self, baseline: str) -> dict:
        agent_dir = str(Path("eval_report_gen/baselines") / baseline / "agent_outputs")
        return run_eval(
            gt_dir=self.gt_dir,
            agent_dir=agent_dir,
            case_ids=self.cases,
            task_config=self.cfg,
        )

    def test_perfect_baseline_scores_perfectly(self):
        report = self._run("perfect")
        self.assertEqual(report["aggregate"]["rating"], "A")
        self.assertAlmostEqual(report["metrics"]["observation_f1"], 1.0, places=4)
        self.assertAlmostEqual(report["metrics"]["report_similarity"], 1.0, places=4)

    def test_constant_normal_baseline_is_valid_but_below_baseline(self):
        report = self._run("constant_normal")
        self.assertEqual(report["aggregate"]["rating"], "C")
        self.assertTrue(report["format"]["output_format_valid"])
        self.assertLess(report["metrics"]["observation_f1"], 0.5)

    def test_empty_baseline_fails(self):
        report = self._run("empty")
        self.assertEqual(report["aggregate"]["rating"], "F")
        self.assertFalse(report["format"]["output_format_valid"])


if __name__ == "__main__":
    unittest.main()
