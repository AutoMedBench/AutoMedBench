#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval_report_gen.api_key_loader import load_api_keys


VALID_KEYS = [
    "sk-first-key",
    "sk-second-key",
    "sk-third-key",
    "sk-fourth-key",
]


class ApiKeyLoaderTests(unittest.TestCase):
    def test_load_api_keys_requires_exactly_four_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "key.txt"
            path.write_text("\n".join(VALID_KEYS[:3]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 4 API keys"):
                load_api_keys(path)

    def test_load_api_keys_rejects_non_key_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "key.txt"
            path.write_text("\n".join(VALID_KEYS[:3] + ["API Key4:"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid content on line\\(s\\): 4"):
                load_api_keys(path)

    def test_load_api_keys_returns_four_keys_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "key.txt"
            path.write_text("\n".join(VALID_KEYS) + "\n", encoding="utf-8")
            self.assertEqual(load_api_keys(path), VALID_KEYS)


if __name__ == "__main__":
    unittest.main()
