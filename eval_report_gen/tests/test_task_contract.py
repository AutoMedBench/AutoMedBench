#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

from eval_report_gen.format_checker import check_submission
from eval_report_gen.task_loader import discover_cases, load_task_config


class TaskContractTests(unittest.TestCase):
    def test_task_config_has_expected_contract(self):
        cfg = load_task_config("mimic-cxr-report-task")
        self.assertEqual(cfg["grouping_unit"], "study")
        self.assertEqual(cfg["output_filename"], "report.txt")
        self.assertTrue(cfg["completion_requires_all_cases"])

    def test_staged_pilot_has_ten_cases(self):
        cases = discover_cases("mimic-cxr-report-task")
        self.assertEqual(len(cases), 10)

    def test_perfect_baseline_passes_format_checker(self):
        cfg = load_task_config("mimic-cxr-report-task")
        cases = discover_cases("mimic-cxr-report-task")
        result = check_submission(
            agent_dir=Path("eval_report_gen/baselines/perfect/agent_outputs"),
            case_ids=cases,
            task_config=cfg,
        )
        self.assertTrue(result["output_format_valid"])
        self.assertEqual(result["completion_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
