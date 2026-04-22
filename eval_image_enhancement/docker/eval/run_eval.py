#!/usr/bin/env python3
"""v5 eval-container runner — black-box scoring of agent outputs.

Reads /agent_outputs/<pid>/enhanced.npy, compares to /data/private/<pid>/reference.npy,
writes /results/detail_report.json with:
  format_check, raw metrics, pass_rate (SSIM ≥ classical - 0.02),
  clinical score, rating, shuffled-negative-control, per-patient details.

Env: PATIENT_IDS (comma), TASK, AGENT_NAME, TIER, REPEAT_IDX
"""

import json
import os
import sys
import time

sys.path.insert(0, "/eval")

from enhancement_scorer import assign_rating, clinical_score, score_all
from format_checker import check_submission


AGENT_NAME  = os.environ.get("AGENT_NAME", "unknown")
TASK        = os.environ.get("TASK", "")
TIER        = os.environ.get("TIER", "lite")
REPEAT_IDX  = os.environ.get("REPEAT_IDX", "0")
PATIENT_IDS = [p for p in os.environ.get("PATIENT_IDS", "").split(",") if p]

AGENT_OUTPUTS_DIR = "/agent_outputs"
PRIVATE_DIR       = "/data/private"
RESULTS_DIR       = "/results"


def load_pass_rate_threshold(task: str) -> tuple[float | None, float | None]:
    """Return (classical_ssim, threshold=classical_ssim-0.02) for the task."""
    bands_path = os.path.join("/eval", task, "baseline_bands.json")
    if not os.path.isfile(bands_path):
        return None, None
    with open(bands_path) as f:
        bands = json.load(f)
    cls_key = bands.get("anchor_keys", {}).get("classical")
    if not cls_key:
        return None, None
    cls_ssim = bands["baselines"][cls_key].get("mean_ssim")
    if cls_ssim is None:
        return None, None
    return float(cls_ssim), float(cls_ssim) - 0.02


def main() -> int:
    t0 = time.time()
    print(f"[eval] task={TASK} agent={AGENT_NAME} tier={TIER} repeat={REPEAT_IDX}")
    print(f"[eval] {len(PATIENT_IDS)} patients")

    report = {
        "agent":       AGENT_NAME,
        "task":        TASK,
        "tier":        TIER,
        "repeat_idx":  REPEAT_IDX,
        "n_patients":  len(PATIENT_IDS),
        "patient_ids": PATIENT_IDS,
    }

    # Format check
    fmt = check_submission(AGENT_OUTPUTS_DIR, PRIVATE_DIR, PATIENT_IDS)
    report["format"] = fmt
    print(f"[eval] format: {fmt['n_valid']}/{fmt['n_patients']} valid "
          f"(all_valid={fmt['output_format_valid']})")

    completion_rate = fmt["n_valid"] / max(1, fmt["n_patients"])

    if fmt["n_valid"] == 0:
        # Nothing to score
        report["rating"] = assign_rating(
            TASK, None, None, None,
            format_valid=False, aborted=False,
            completion_rate=completion_rate,
        )
        report["scores"]  = None
        report["shuffle"] = None
        report["eval_elapsed_s"] = round(time.time() - t0, 2)
        _write_report(report)
        return 0

    # Raw scoring (aligned)
    s_raw = score_all(AGENT_OUTPUTS_DIR, PRIVATE_DIR, PATIENT_IDS, shuffle=False)
    report["scores"] = s_raw
    print(f"[eval] raw: PSNR={s_raw['mean_psnr']:.2f} SSIM={s_raw['mean_ssim']:.4f} "
          f"LPIPS={s_raw['mean_lpips']:.4f}")

    # Shuffled negative control — confirms scorer is not returning high numbers by default
    s_shuf = score_all(AGENT_OUTPUTS_DIR, PRIVATE_DIR, PATIENT_IDS, shuffle=True)
    report["shuffle"] = {
        "mean_psnr":  s_shuf["mean_psnr"],
        "mean_ssim":  s_shuf["mean_ssim"],
        "mean_lpips": s_shuf["mean_lpips"],
    }
    print(f"[eval] shuffle-NC: PSNR={s_shuf['mean_psnr']:.2f} "
          f"SSIM={s_shuf['mean_ssim']:.4f}")

    # Clinical (per-task normalized) score + rating
    cs = clinical_score(s_raw["mean_psnr"], s_raw["mean_ssim"],
                        s_raw["mean_lpips"], task=TASK)
    report["clinical"] = cs

    rating = assign_rating(
        TASK, s_raw["mean_psnr"], s_raw["mean_ssim"], s_raw["mean_lpips"],
        format_valid=fmt["output_format_valid"],
        aborted=False, completion_rate=completion_rate,
    )
    report["rating"] = rating

    # Pass rate: per-case SSIM ≥ classical_baseline_SSIM − 0.02
    cls_ssim, threshold = load_pass_rate_threshold(TASK)
    if threshold is not None:
        per = s_raw.get("per_patient") or []
        passes = [p for p in per if (p.get("ssim") or 0) >= threshold]
        pass_rate = len(passes) / max(1, len(per))
        report["pass_rate"] = {
            "threshold_ssim": round(threshold, 4),
            "classical_ssim": round(cls_ssim, 4),
            "n_passed":       len(passes),
            "n_scored":       len(per),
            "pass_rate":      round(pass_rate, 4),
        }
        print(f"[eval] pass_rate = {pass_rate:.3f} ({len(passes)}/{len(per)})  "
              f"thr={threshold:.4f}")

    report["eval_elapsed_s"] = round(time.time() - t0, 2)
    print(f"[eval] rating={rating.get('rating','?')} clinical={cs['clinical']:.4f} "
          f"elapsed={report['eval_elapsed_s']:.1f}s")

    _write_report(report)
    return 0


def _write_report(report: dict) -> None:
    out = os.path.join(RESULTS_DIR, "detail_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
