#!/usr/bin/env python3
"""Compute dataset-calibrated rating bands by running reference baselines.

For each task (LDCT, MRI-SR) on the staged v3 100-patient set:
  1. Run 4 baselines: identity, classical (BM3D / bicubic), DNN (DRUNet / SwinIR2x), perfect
  2. Compute mean PSNR + SSIM per baseline
  3. Write <task>/baseline_bands.json with A/B thresholds anchored on the DNN baseline:
       A = (DNN_ssim + 0.02, DNN_psnr + 1.0)   → better than specified Lite DNN
       B = (DNN_ssim - 0.02, DNN_psnr - 1.0)   → within 1 dB / 0.02 SSIM of DNN
       C = (classical_ssim - 0.01, classical_psnr - 1.0)   → below classical floor
       F = completion < 50% / aborted

Design rationale documented in refine-logs/SCORING_LITERATURE_REVIEW.md.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))
from enhancement_scorer import psnr, ssim, lpips_score  # noqa: E402


def _norm_hu_to_01(x):
    """Map CT HU in [-1024, +3072] to [0, 1] for DNN input."""
    return np.clip((x - (-1024.0)) / (3072.0 - (-1024.0)), 0.0, 1.0).astype(np.float32)


def _invert_01_to_hu(y):
    return (y * (3072.0 - (-1024.0)) + (-1024.0)).astype(np.float32)


# ==========================================================================
# LDCT baselines
# ==========================================================================

def ldct_identity(x):
    return x.astype(np.float32)


def ldct_gaussian(x):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x.astype(np.float32), sigma=1.2)


def ldct_bm3d(x):
    import bm3d
    from skimage.restoration import estimate_sigma
    x01 = _norm_hu_to_01(x)
    sig = float(estimate_sigma(x01))
    y01 = bm3d.bm3d(x01, sigma_psd=sig).astype(np.float32)
    return _invert_01_to_hu(y01)


_DRUNET_MODEL = None


def ldct_drunet(x):
    """DNN baseline for LDCT — DRUNet via deepinv. GPU required."""
    global _DRUNET_MODEL
    import torch
    import deepinv as dinv
    if _DRUNET_MODEL is None:
        _DRUNET_MODEL = dinv.models.DRUNet(in_channels=1, out_channels=1,
                                            pretrained="download").eval().cuda()
    x01 = _norm_hu_to_01(x)
    t = torch.from_numpy(x01).unsqueeze(0).unsqueeze(0).cuda()
    from skimage.restoration import estimate_sigma
    sig = float(estimate_sigma(x01))
    with torch.no_grad():
        y01 = _DRUNET_MODEL(t, sigma=sig).squeeze().cpu().numpy().astype(np.float32)
    return _invert_01_to_hu(y01)


def ldct_perfect(x, ref):
    return ref.astype(np.float32)


# ==========================================================================
# MRI-SR baselines
# ==========================================================================

def mri_identity_up(lr, target_shape):
    import torch, torch.nn.functional as F
    t = torch.from_numpy(lr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=target_shape, mode="nearest")
    return up.squeeze().numpy().astype(np.float32)


def mri_gaussian_up(lr, target_shape):
    from scipy.ndimage import gaussian_filter
    import torch, torch.nn.functional as F
    blurred = gaussian_filter(lr.astype(np.float32), sigma=1.2)
    t = torch.from_numpy(blurred).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=target_shape, mode="bicubic", align_corners=False)
    return up.squeeze().numpy().astype(np.float32)


def mri_bicubic(lr, target_shape):
    import torch, torch.nn.functional as F
    t = torch.from_numpy(lr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=target_shape, mode="bicubic", align_corners=False)
    return up.squeeze().numpy().astype(np.float32)


_SWINIR_MODEL = None
_SWINIR_PROC = None


def mri_swinir(lr, target_shape):
    """DNN baseline for MRI-SR — Swin2SR 2× via HuggingFace. GPU required."""
    global _SWINIR_MODEL, _SWINIR_PROC
    import torch
    if _SWINIR_MODEL is None:
        from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
        mdl = "caidas/swin2SR-classical-sr-x2-64"
        _SWINIR_PROC = AutoImageProcessor.from_pretrained(mdl)
        _SWINIR_MODEL = Swin2SRForImageSuperResolution.from_pretrained(mdl).eval().cuda()
    from PIL import Image
    # lr is [0,1] float32, convert to uint8 RGB
    lr_u8 = (np.clip(lr, 0.0, 1.0) * 255).astype(np.uint8)
    pil = Image.fromarray(lr_u8, mode="L").convert("RGB")
    inputs = _SWINIR_PROC(pil, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = _SWINIR_MODEL(**inputs).reconstruction
    # Output is (1, 3, H', W') in [0,1] pixel-value range (after processor normalizes)
    out_np = out.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)  # collapse RGB → grayscale mean
    out_np = np.clip(out_np, 0.0, 1.0)
    # Resize to target shape if needed (Swin2SR may crop slightly)
    if out_np.shape != target_shape:
        import torch.nn.functional as F
        t = torch.from_numpy(out_np).unsqueeze(0).unsqueeze(0)
        t2 = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
        out_np = t2.squeeze().numpy().astype(np.float32)
    return out_np


# ==========================================================================
# Driver
# ==========================================================================

TASK_CONFIG = {
    "ldct": {
        "data_dir": "LDCT_SimNICT",
        "baselines": {
            "identity": lambda x, ref: ldct_identity(x),
            "gaussian": lambda x, ref: ldct_gaussian(x),
            "classical_bm3d": lambda x, ref: ldct_bm3d(x),
            "dnn_drunet":     lambda x, ref: ldct_drunet(x),
            "perfect":        lambda x, ref: ldct_perfect(x, ref),
        },
        "classical_key": "classical_bm3d",
        "dnn_key":       "dnn_drunet",
    },
    "mri-sr": {
        "data_dir": "MRI_SR_SRMRI",
        "baselines": {
            "identity_nearest_up": lambda x, ref: mri_identity_up(x, ref.shape),
            "gaussian_bicubic":    lambda x, ref: mri_gaussian_up(x, ref.shape),
            "classical_bicubic":   lambda x, ref: mri_bicubic(x, ref.shape),
            "dnn_swinir_x2":       lambda x, ref: mri_swinir(x, ref.shape),
            "perfect":             lambda x, ref: ref.astype(np.float32),
        },
        "classical_key": "classical_bicubic",
        "dnn_key":       "dnn_swinir_x2",
    },
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_CONFIG))
    p.add_argument("--data-root",
                   default=str(Path(__file__).resolve().parent.parent.parent / "data"))
    p.add_argument("--output", default=None,
                   help="where to write baseline_bands.json; default = <task_dir>/baseline_bands.json")
    p.add_argument("--limit", type=int, default=None, help="optional patient cap for fast debug")
    args = p.parse_args()

    cfg = TASK_CONFIG[args.task]
    data_task = Path(args.data_root) / cfg["data_dir"]
    public_dir = data_task / "public"
    priv_dir = data_task / "private"
    pids = sorted([p.name for p in public_dir.iterdir() if p.is_dir()])
    if args.limit:
        pids = pids[: args.limit]
    print(f"computing baselines on {len(pids)} patients, task={args.task}")

    # Pre-load all inputs + refs so baselines score on same set
    inputs = {pid: np.load(public_dir / pid / "input.npy") for pid in pids}
    refs = {pid: np.load(priv_dir / pid / "reference.npy") for pid in pids}
    data_ranges = {pid: float(r.max() - r.min()) for pid, r in refs.items()}

    results: dict = {"task": args.task, "n_patients": len(pids), "baselines": {}}
    for name, fn in cfg["baselines"].items():
        print(f"\n  [{name}] ...")
        t0 = time.time()
        psnrs, ssims, lpipss = [], [], []
        for pid in pids:
            inp = inputs[pid]
            ref = refs[pid]
            out = fn(inp, ref)
            if out.shape != ref.shape:
                raise ValueError(f"{name} output {out.shape} != ref {ref.shape}")
            dr = data_ranges[pid]
            psnrs.append(psnr(out, ref, data_range=dr))
            ssims.append(ssim(out, ref, data_range=dr))
            lpipss.append(lpips_score(out, ref))
        dt = time.time() - t0
        print(f"  [{name}] done in {dt:.1f}s  "
              f"PSNR={np.mean(psnrs):.3f}  SSIM={np.mean(ssims):.4f}  LPIPS={np.mean(lpipss):.4f}")
        results["baselines"][name] = {
            "mean_psnr":  float(np.mean(psnrs)),
            "mean_ssim":  float(np.mean(ssims)),
            "mean_lpips": float(np.mean(lpipss)),
            "time_s":     round(dt, 2),
            "n":          len(pids),
        }

    classical = results["baselines"][cfg["classical_key"]]
    dnn = results["baselines"][cfg["dnn_key"]]

    # Pick the STRONGER of {classical, dnn} as the top baseline anchor.
    # If the specified DNN transfers poorly (e.g., Swin2SR on grayscale MRI),
    # classical wins and A-band is defined vs classical so "good" stays meaningful.
    if dnn["mean_ssim"] >= classical["mean_ssim"]:
        top_key, top = cfg["dnn_key"], dnn
        floor_key, floor = cfg["classical_key"], classical
    else:
        top_key, top = cfg["classical_key"], classical
        floor_key, floor = cfg["dnn_key"], dnn

    # A = beats stronger baseline by +0.02 SSIM / +1 dB PSNR
    # B = within ±0.02 SSIM / ±1 dB of stronger baseline
    # C = within ±0.02 SSIM / ±1 dB of weaker baseline (or below B but not F)
    # F = below weaker baseline - 1 dB / completion < 50% / aborted
    results["bands"] = {
        "A": {
            "ssim_min": round(top["mean_ssim"] + 0.02, 4),
            "psnr_min": round(top["mean_psnr"] + 1.0, 3),
            "note": f"beats '{top_key}' baseline by +0.02 SSIM and +1 dB PSNR",
        },
        "B": {
            "ssim_min": round(top["mean_ssim"] - 0.02, 4),
            "psnr_min": round(top["mean_psnr"] - 1.0, 3),
            "note": f"within ±0.02 SSIM and ±1 dB of '{top_key}' (stronger) baseline",
        },
        "C": {
            # C band must be meaningfully distinct from B. When classical ≈ DNN
            # (as in LDCT, both ~41.8 dB), a naive "floor - 1 dB" collapses
            # onto B. Anchor C to B minus a fixed margin instead.
            "ssim_min": round(top["mean_ssim"] - 0.02 - 0.03, 4),
            "psnr_min": round(top["mean_psnr"] - 1.0 - 2.0, 3),
            "note": f"approaches '{top_key}' baseline but more than 3 dB / 0.05 SSIM below",
        },
        "F": {
            "note": f"below '{floor_key}' (weaker) baseline or <50% format valid or aborted",
        },
    }
    results["anchor_keys"] = {
        "stronger": top_key, "weaker": floor_key,
        "classical": cfg["classical_key"], "dnn": cfg["dnn_key"],
    }

    out_path = Path(args.output) if args.output else (THIS_DIR / f"{args.task}-denoising-task" if args.task == "ldct" else THIS_DIR / "mri-sr-task") / "baseline_bands.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_path}")
    print(f"\n=== {args.task.upper()} RATING BANDS (dataset-calibrated) ===")
    print(f"  A requires: SSIM ≥ {results['bands']['A']['ssim_min']:.4f}  AND  PSNR ≥ {results['bands']['A']['psnr_min']:.3f}")
    print(f"  B requires: SSIM ≥ {results['bands']['B']['ssim_min']:.4f}  AND  PSNR ≥ {results['bands']['B']['psnr_min']:.3f}")
    print(f"  C requires: SSIM ≥ {results['bands']['C']['ssim_min']:.4f}  AND  PSNR ≥ {results['bands']['C']['psnr_min']:.3f}")
    print(f"  F: below C / fmt < 50% / aborted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
