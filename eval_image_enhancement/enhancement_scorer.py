"""PSNR + SSIM + LPIPS scorer for 2D medical image enhancement.

Canonical 2-layer data access:
  predictions -> <output_dir>/agents_outputs/<pid>/enhanced.npy
  references  -> data/<DatasetName>/private/<pid>/reference.npy
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


_LPIPS_SINGLETON = None


def _load_lpips(net: str = "alex"):
    global _LPIPS_SINGLETON
    if _LPIPS_SINGLETON is not None:
        return _LPIPS_SINGLETON
    import torch
    import lpips
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    _LPIPS_SINGLETON = (model, device, torch)
    return _LPIPS_SINGLETON


def _as_float32_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def psnr(pred: np.ndarray, ref: np.ndarray, data_range: float) -> float:
    pred = _as_float32_2d(pred)
    ref = _as_float32_2d(ref)
    mse = float(np.mean((pred - ref) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(pred: np.ndarray, ref: np.ndarray, data_range: float) -> float:
    from skimage.metrics import structural_similarity
    pred = _as_float32_2d(pred)
    ref = _as_float32_2d(ref)
    return float(structural_similarity(ref, pred, data_range=data_range,
                                       gaussian_weights=True,
                                       use_sample_covariance=False))


def lpips_score(pred: np.ndarray, ref: np.ndarray) -> float:
    """LPIPS — lower is better. Normalizes each input to [-1,1] independently."""
    model, device, torch = _load_lpips()
    pred = _as_float32_2d(pred)
    ref = _as_float32_2d(ref)

    def _norm(x):
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return ((x - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)

    p_t = torch.from_numpy(_norm(pred)).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    r_t = torch.from_numpy(_norm(ref)).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        return float(model(p_t, r_t).squeeze().item())


@dataclass
class PerPatientScore:
    patient_id: str
    psnr: float
    ssim: float
    lpips: float
    data_range: float


def _load_data_ranges(private_dir: str) -> dict[str, float]:
    """Read data_range per patient from ground_truth.csv; fall back to reference min/max."""
    out: dict[str, float] = {}
    csv_path = os.path.join(private_dir, "ground_truth.csv")
    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if "data_range" in row and row["data_range"]:
                    try:
                        out[row["patient_id"]] = float(row["data_range"])
                    except ValueError:
                        pass
    return out


def score_all(agent_outputs_dir: str, private_dir: str, patient_ids: list[str],
              shuffle: bool = False) -> dict[str, Any]:
    refs: dict[str, np.ndarray] = {}
    ranges = _load_data_ranges(private_dir)
    for pid in patient_ids:
        refs[pid] = np.load(os.path.join(private_dir, pid, "reference.npy"))
        if pid not in ranges:
            ranges[pid] = float(refs[pid].max() - refs[pid].min())

    per_patient: list[PerPatientScore] = []
    for idx, pid in enumerate(patient_ids):
        pred_path = os.path.join(agent_outputs_dir, pid, "enhanced.npy")
        if not os.path.isfile(pred_path):
            per_patient.append(PerPatientScore(pid, float("nan"), float("nan"),
                                               float("nan"), float("nan")))
            continue
        pred = np.load(pred_path)
        ref_pid = patient_ids[(idx + 1) % len(patient_ids)] if shuffle else pid
        ref = refs[ref_pid]
        dr = ranges[ref_pid]
        per_patient.append(PerPatientScore(
            patient_id=pid,
            psnr=psnr(pred, ref, data_range=dr),
            ssim=ssim(pred, ref, data_range=dr),
            lpips=lpips_score(pred, ref),
            data_range=dr,
        ))

    def _mean(attr: str) -> float:
        vals = [getattr(p, attr) for p in per_patient if not np.isnan(getattr(p, attr))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "mean_psnr": _mean("psnr"),
        "mean_ssim": _mean("ssim"),
        "mean_lpips": _mean("lpips"),
        "per_patient": [p.__dict__ for p in per_patient],
        "n_patients": len(per_patient),
        "n_valid": sum(1 for p in per_patient if not np.isnan(p.psnr)),
        "shuffle": shuffle,
    }


# ---------------------------------------------------------------------------
# Per-task score normalization (v2)
#
# Current literature-grounded ranges for linear normalization:
#
#   LDCT denoising:
#     PSNR  — typical DNN range 30–45 dB (RED-CNN 40, MAP-NN 42, DU-GAN 44)
#             AAPM LDCT Challenge baselines ~30 dB; SOTA DNN ~45 dB
#     SSIM  — [0, 1] direct
#     LPIPS — typical 0.02–0.30 for restoration baselines
#
#   MRI super-resolution (2×):
#     PSNR  — typical range 20–35 dB (SwinIR 30-34, ArSSR 28-30, bicubic ~24)
#     SSIM  — [0, 1] direct
#     LPIPS — typical 0.10–0.50 for bicubic, 0.05–0.20 for DNN
#
# These windows spread realistic method scores across [0, 1] cleanly. They
# are per-task because LDCT HU range and MRI [0,1] range give systematically
# different PSNR regimes.
# ---------------------------------------------------------------------------

TASK_NORM: dict[str, dict[str, float]] = {
    "ldct-denoising-task": {
        "psnr_low":  30.0, "psnr_high": 45.0,
        "lpips_max": 0.30,
    },
    "mri-sr-task": {
        "psnr_low":  20.0, "psnr_high": 35.0,
        "lpips_max": 0.50,
    },
}

# Per-task A / B / C thresholds on RAW metrics (literature-aligned).
#   A = good (meets or exceeds strong DNN level)
#   B = okay (meets or exceeds classical baseline level)
#   C = below classical
#   Thresholds combined with AND logic on (PSNR, SSIM, LPIPS).
RATING_THRESHOLDS: dict[str, dict[str, dict[str, float]]] = {
    "ldct-denoising-task": {
        "A": {"psnr": 42.0, "ssim": 0.95, "lpips": 0.05},
        "B": {"psnr": 38.0, "ssim": 0.90, "lpips": 0.10},
    },
    "mri-sr-task": {
        "A": {"psnr": 30.0, "ssim": 0.85, "lpips": 0.15},
        "B": {"psnr": 26.0, "ssim": 0.70, "lpips": 0.25},
    },
}


def _norm_psnr(psnr: float, lo: float, hi: float) -> float:
    return float(np.clip((psnr - lo) / (hi - lo), 0.0, 1.0))


def _norm_lpips(lpips: float, lpips_max: float) -> float:
    # LPIPS low is good. 0 → 1.0, lpips_max → 0.0
    return float(np.clip(1.0 - (lpips / lpips_max), 0.0, 1.0))


def clinical_score(
    mean_psnr: float,
    mean_ssim: float,
    mean_lpips: float,
    task: str | None = None,
) -> dict[str, float]:
    """Per-task normalized clinical score.

    If `task` is one of TASK_NORM keys, use per-task linear windows for PSNR
    and LPIPS (literature-aligned). Otherwise fall back to the v1 generic
    window for backward compatibility.
    """
    if task in TASK_NORM:
        nc = TASK_NORM[task]
        psnr_norm = _norm_psnr(mean_psnr, nc["psnr_low"], nc["psnr_high"])
        lpips_norm = _norm_lpips(mean_lpips, nc["lpips_max"])
    else:
        # v1 fallback (generic) — kept so old runs still parse
        psnr_norm = float(np.clip((mean_psnr - 20.0) / 30.0, 0.0, 1.0))
        lpips_norm = float(np.clip(1.0 - mean_lpips, 0.0, 1.0))

    ssim_norm = float(np.clip(mean_ssim, 0.0, 1.0))
    return {
        "task": task or "generic",
        "psnr_norm": round(psnr_norm, 4),
        "ssim_norm": round(ssim_norm, 4),
        "lpips_norm": round(lpips_norm, 4),
        "clinical": round((psnr_norm + ssim_norm + lpips_norm) / 3.0, 4),
    }


def _load_bands(task: str) -> dict | None:
    """Load dataset-calibrated bands from <task_dir>/baseline_bands.json if present (v3)."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    task_folder = task if task.endswith("-task") else f"{task}-task"
    p = os.path.join(here, task_folder, "baseline_bands.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return None


def assign_rating(
    task: str,
    mean_psnr: float | None,
    mean_ssim: float | None,
    mean_lpips: float | None,
    format_valid: bool,
    aborted: bool = False,
    completion_rate: float | None = None,
) -> dict[str, Any]:
    """A/B/C/F rating. v3: dataset-calibrated via baseline_bands.json if present;
    otherwise v2 literature-hardcoded fallback.

    v3 logic:
      A  beats the specified DNN baseline on PSNR+SSIM
      B  matches DNN baseline (within 1dB / 0.02 SSIM)
      C  matches classical baseline (within 1dB / 0.02 SSIM)
      F  below classical / completion < 50% / aborted

    LPIPS is reported, not gated (Breger 2025, Lee 2025).
    """
    if aborted or not format_valid or mean_psnr is None:
        return {"rating": "F", "tier_name": "fail", "reasons": ["invalid_or_aborted"], "source": "hard_fail"}
    if completion_rate is not None and completion_rate < 0.5:
        return {"rating": "F", "tier_name": "fail",
                "reasons": [f"completion_rate={completion_rate:.2f}<0.5"], "source": "hard_fail"}

    bands = _load_bands(task)
    if bands:
        def meets(level):
            t = bands["bands"][level]
            fails = []
            if mean_psnr < t["psnr_min"]:
                fails.append(f"psnr={mean_psnr:.2f}<{t['psnr_min']}")
            if mean_ssim < t["ssim_min"]:
                fails.append(f"ssim={mean_ssim:.4f}<{t['ssim_min']}")
            return (len(fails) == 0), fails
        for lvl in ["A", "B", "C"]:
            ok, fails = meets(lvl)
            if ok:
                return {"rating": lvl, "reasons": fails, "source": "v3_dataset_calibrated",
                        "tier_name": {"A": "good", "B": "matches_dnn", "C": "matches_classical"}[lvl]}
        return {"rating": "F", "tier_name": "below_classical",
                "reasons": [f"ssim={mean_ssim:.4f}", f"psnr={mean_psnr:.2f}"],
                "source": "v3_dataset_calibrated"}

    th = RATING_THRESHOLDS.get(task)
    if th is None:
        return {"rating": "C", "tier_name": "unknown_task", "reasons": ["no_thresholds_for_task"]}

    def meets(level: str) -> tuple[bool, list[str]]:
        t = th[level]
        fails = []
        if mean_psnr < t["psnr"]:
            fails.append(f"psnr={mean_psnr:.2f}<{t['psnr']}")
        if mean_ssim < t["ssim"]:
            fails.append(f"ssim={mean_ssim:.4f}<{t['ssim']}")
        if mean_lpips > t["lpips"]:
            fails.append(f"lpips={mean_lpips:.4f}>{t['lpips']}")
        return (len(fails) == 0), fails

    ok_a, fails_a = meets("A")
    if ok_a:
        return {"rating": "A", "tier_name": "good", "reasons": []}
    ok_b, fails_b = meets("B")
    if ok_b:
        return {"rating": "B", "tier_name": "okay", "reasons": fails_a}
    return {"rating": "C", "tier_name": "below_baseline", "reasons": fails_b}
