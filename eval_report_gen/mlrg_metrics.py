#!/usr/bin/env python3
"""Optional MLRG-backed report metrics for eval_report_gen."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _validate_path(path_value: str | None, label: str, must_exist: bool = True) -> str:
    if not path_value:
        raise ValueError(f"Missing required MLRG setting: {label}")
    path = Path(path_value)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def compute_mlrg_scores(
    references: list[str],
    predictions: list[str],
    task_config: dict,
) -> dict:
    repo_path = _validate_path(task_config.get("mlrg_repo_path"), "mlrg_repo_path")
    repo_root = Path(repo_path).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # `radgraph`'s vendored AllenNLP stack is brittle on Python 3.12 when the
    # `overrides_` package performs bytecode inspection. A no-op decorator is
    # sufficient for these metrics, so install a lightweight shim before
    # importing MLRG's metric module.
    if "overrides_" not in sys.modules:
        shim = types.ModuleType("overrides_")

        def _decorator(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def _wrap(func):
                return func

            return _wrap

        shim.overrides = _decorator
        sys.modules["overrides_"] = shim
    if "overrides" not in sys.modules:
        shim = types.ModuleType("overrides")
        shim.overrides = sys.modules["overrides_"].overrides
        sys.modules["overrides"] = shim

    try:
        from transformers import tokenization_utils_base as _tub
    except Exception:
        _tub = None
    if _tub is not None and not getattr(_tub, "_medagentsbench_patched", False):
        _orig_init = _tub.PreTrainedTokenizerBase.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs.pop("add_special_tokens", None)
            return _orig_init(self, *args, **kwargs)

        _tub.PreTrainedTokenizerBase.__init__ = _patched_init
        _tub._medagentsbench_patched = True

    try:
        from tools.metrics.metrics import compute_all_scores
    except ImportError as exc:
        raise ImportError(
            "Failed to import MLRG metrics. Install MLRG requirements and ensure "
            f"the repo is available at {repo_root}."
        ) from exc

    args = {
        "chexbert_path": _validate_path(task_config.get("chexbert_path"), "chexbert_path"),
        "bert_path": _validate_path(task_config.get("bert_path"), "bert_path"),
        "radgraph_path": _validate_path(task_config.get("radgraph_path"), "radgraph_path"),
    }
    scores = compute_all_scores(references, predictions, args)

    bleu_values = [
        float(scores.get("BLEU_1", 0.0)),
        float(scores.get("BLEU_2", 0.0)),
        float(scores.get("BLEU_3", 0.0)),
        float(scores.get("BLEU_4", 0.0)),
    ]
    scores["BLEU"] = round(sum(bleu_values) / len(bleu_values), 4)
    scores["micro_average_precision"] = round(float(scores.get("chexbert_all_micro_p", 0.0)), 4)
    scores["micro_average_recall"] = round(float(scores.get("chexbert_all_micro_r", 0.0)), 4)
    scores["micro_average_f1"] = round(float(scores.get("chexbert_all_micro_f1", 0.0)), 4)
    scores["F1RadGraph"] = round(float(scores.get("F1-Radgraph-partial", 0.0)), 4)

    normalized = {}
    for key, value in scores.items():
        if isinstance(value, (int, float)):
            normalized[key] = round(float(value), 4)
        else:
            normalized[key] = value
    return normalized
