#!/usr/bin/env python3
from __future__ import annotations

import unittest

from eval_report_gen.report_scorer import extract_findings_text, extract_observations


class ReportScorerTests(unittest.TestCase):
    def test_combined_negation_does_not_trigger_pneumothorax(self):
        labels = extract_observations(
            "No focal airspace consolidation, pleural effusion, or pneumothorax."
        )
        self.assertEqual(labels["consolidation"], 0)
        self.assertEqual(labels["pleural_effusion"], 0)
        self.assertEqual(labels["pneumothorax"], 0)

    def test_plural_pneumothoraces_negation_is_recognized(self):
        labels = extract_observations("No pleural effusions or pneumothoraces.")
        self.assertEqual(labels["pleural_effusion"], 0)
        self.assertEqual(labels["pneumothorax"], 0)

    def test_no_acute_cardiopulmonary_abnormality_is_recognized(self):
        labels = extract_observations("No acute cardiopulmonary abnormality.")
        self.assertEqual(labels["no_acute_process"], 1)

    def test_extract_findings_text_prefers_findings_section(self):
        text = (
            "FINAL REPORT\n"
            "FINDINGS: Lungs are clear. No pleural effusion.\n"
            "IMPRESSION: No acute cardiopulmonary process.\n"
        )
        self.assertEqual(
            extract_findings_text(text),
            "Lungs are clear. No pleural effusion.",
        )

    def test_extract_findings_text_raises_when_strict_and_missing(self):
        with self.assertRaises(ValueError):
            extract_findings_text("IMPRESSION: No acute cardiopulmonary process.", strict=True)


if __name__ == "__main__":
    unittest.main()
