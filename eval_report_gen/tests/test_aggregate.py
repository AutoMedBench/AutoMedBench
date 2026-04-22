#!/usr/bin/env python3
from __future__ import annotations

import unittest

from eval_report_gen.aggregate import compute_clinical_score


class AggregateTests(unittest.TestCase):
    def test_compute_clinical_score_uses_named_metric_weights(self):
        metrics = {
            "BLEU": 0.2,
            "METEOR": 0.3,
            "ROUGE_L": 0.4,
            "F1RadGraph": 0.5,
            "micro_average_precision": 0.6,
            "micro_average_recall": 0.7,
            "micro_average_f1": 0.8,
        }
        weights = {key: 1 / 7 for key in metrics}
        score = compute_clinical_score(metrics, weights)
        self.assertAlmostEqual(score, 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
