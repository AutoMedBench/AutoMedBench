#!/usr/bin/env python3
"""Tests for Docker mount separation in eval_report_gen."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval_report_gen.docker.orchestrator import agent_run_cmd, eval_run_cmd


class OrchestratorTests(unittest.TestCase):
    def test_agent_container_does_not_mount_private_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = agent_run_cmd(tmp, "mimic-cxr-report-task", "lite", "baseline-perfect")
        joined = " ".join(cmd)
        self.assertIn("/data/public:ro", joined)
        self.assertNotIn("/data/private:ro", joined)

    def test_eval_container_mounts_private_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = eval_run_cmd(tmp, "mimic-cxr-report-task", "baseline-perfect")
        joined = " ".join(cmd)
        self.assertIn("/data/public:ro", joined)
        self.assertIn("/data/private:ro", joined)
        self.assertIn("/agent_outputs:ro", joined)


if __name__ == "__main__":
    unittest.main()
