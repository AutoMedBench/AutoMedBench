#!/usr/bin/env python3
"""v5 matrix figures: pass-rate heatmap, per-task PSNR/SSIM scatter with baseline bands,
efficiency scatter. Reads runs/matrix_v5*/summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


AGENTS_SHORT = ["opus46", "gpt54", "gemini31pro", "glm5", "minimax25", "qwen35"]
AGENT_LABEL = {
    "opus46":      "Claude Opus 4.6",
    "gpt54":       "GPT-5.4",
    "gemini31pro": "Gemini 3.1 Pro",
    "glm5":        "GLM-5",
    "minimax25":   "MiniMax 2.5",
    "qwen35":      "Qwen 3.5",
}
AGENT_COLOR = {
    "opus46":      "#D97757",
    "gpt54":       "#10A37F",
    "gemini31pro": "#4285F4",
    "glm5":        "#7B5BC5",
    "minimax25":   "#E85C3A",
    "qwen35":      "#F0A202",
}
TIERS = ["lite", "standard"]
TASKS = [
    ("ldct-denoising-task", "LDCT Denoising"),
    ("mri-sr-task",         "MRI SR ×2"),
]
CELLS = [
    ("ldct-denoising-task", "lite",     "LDCT / lite"),
    ("ldct-denoising-task", "standard", "LDCT / std"),
    ("mri-sr-task",         "lite",     "MRI-SR / lite"),
    ("mri-sr-task",         "standard", "MRI-SR / std"),
]
IE_DIR = Path(__file__).resolve().parent.parent


def load_bands(task: str) -> dict:
    return json.load(open(IE_DIR / task / "baseline_bands.json"))


def index_cells(summary: dict) -> dict:
    return {(c["task"], c["tier"], c["agent"]): c for c in summary["cells"]}


def plot_pass_rate_heatmap(summary: dict, out: Path) -> None:
    cells = index_cells(summary)
    M = np.full((len(AGENTS_SHORT), len(CELLS)), np.nan)
    E = np.full((len(AGENTS_SHORT), len(CELLS)), np.nan)
    TXT = [[""] * len(CELLS) for _ in AGENTS_SHORT]
    for i, a in enumerate(AGENTS_SHORT):
        for j, (t, ti, _) in enumerate(CELLS):
            c = cells.get((t, ti, a), {})
            pr = c.get("pass_rate_mean")
            ps = c.get("pass_rate_std") or 0
            if pr is not None:
                M[i, j] = pr
                E[i, j] = ps
                ratings = "".join(c.get("ratings") or [])
                TXT[i][j] = f"{pr:.2f}\n±{ps:.2f}\n{ratings}"
            else:
                TXT[i][j] = "--"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(AGENTS_SHORT)):
        for j in range(len(CELLS)):
            ax.text(j, i, TXT[i][j], ha="center", va="center", fontsize=9,
                    color="black" if M[i, j] > 0.4 else "white")
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([c[2] for c in CELLS], fontsize=10)
    ax.set_yticks(range(len(AGENTS_SHORT)))
    ax.set_yticklabels([AGENT_LABEL[a] for a in AGENTS_SHORT], fontsize=10)
    ax.set_title("Pass rate = per-case SSIM ≥ classical − 0.02  (mean ± std over 3 repeats)\n"
                 "Annotation: pass_rate / std / ratings across repeats",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("pass_rate")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_psnr_ssim_scatter(summary: dict, task: str, task_label: str, out: Path) -> None:
    bands = load_bands(task)
    cells = index_cells(summary)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, tier in zip(axes, TIERS):
        # Band background shading (A/B/C/F regions)
        A = bands["bands"]["A"]
        B = bands["bands"]["B"]
        ax.axhspan(A["ssim_min"], 1.0, xmin=0, xmax=1, color="#c9ead3", alpha=0.35, label="_A")
        # Draw vertical dashed at PSNR mins for visual bands
        ax.axvline(A["psnr_min"], color="#2ca02c", ls="--", lw=1.2, alpha=0.8, label="A band min")
        ax.axvline(B["psnr_min"], color="#ff7f0e", ls="--", lw=1.2, alpha=0.8, label="B band min")
        ax.axhline(A["ssim_min"], color="#2ca02c", ls=":", lw=1)
        ax.axhline(B["ssim_min"], color="#ff7f0e", ls=":", lw=1)

        # Baselines
        for bname, bmetrics in bands["baselines"].items():
            if bname == "perfect":
                continue
            ax.scatter(bmetrics["mean_psnr"], bmetrics["mean_ssim"],
                       marker="x", s=80, color="black", linewidths=2)
            ax.annotate(bname, (bmetrics["mean_psnr"], bmetrics["mean_ssim"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7,
                        color="black", alpha=0.7)

        # Agent means with per-repeat scatter
        for a in AGENTS_SHORT:
            c = cells.get((task, tier, a), {})
            per = c.get("per_repeat") or []
            if not per:
                continue
            xs = [r.get("mean_psnr") for r in per if r.get("mean_psnr") is not None]
            ys = [r.get("mean_ssim") for r in per if r.get("mean_ssim") is not None]
            if not xs:
                continue
            ax.scatter(xs, ys, s=60, color=AGENT_COLOR[a], alpha=0.55,
                       edgecolors="white", linewidths=0.7)
            mx, my = np.mean(xs), np.mean(ys)
            ax.scatter(mx, my, s=150, color=AGENT_COLOR[a], edgecolors="black",
                       linewidths=1.4, label=AGENT_LABEL[a])

        ax.set_title(f"{task_label} — {tier}")
        ax.set_xlabel("mean PSNR (dB)")
        if tier == "lite":
            ax.set_ylabel("mean SSIM")
        ax.grid(True, alpha=0.25)

    # Single legend on the right
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    dedup = [(h, l) for h, l in zip(handles, labels)
             if l not in seen and not seen.add(l)]
    fig.legend([h for h, _ in dedup], [l for _, l in dedup],
               loc="center right", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=9)

    # Adjust xlim to focus on agent region
    all_psnr = []
    all_ssim = []
    for tier in TIERS:
        for a in AGENTS_SHORT:
            c = cells.get((task, tier, a), {})
            for r in c.get("per_repeat") or []:
                if r.get("mean_psnr") is not None:
                    all_psnr.append(r["mean_psnr"])
                if r.get("mean_ssim") is not None:
                    all_ssim.append(r["mean_ssim"])
    if all_psnr:
        for ax in axes:
            ax.set_xlim(min(all_psnr) - 2, max(all_psnr) + 2)
            ax.set_ylim(max(0, min(all_ssim) - 0.05), 1.0)

    fig.subplots_adjust(right=0.85)
    fig.tight_layout(rect=(0, 0, 0.85, 1.0))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(summary: dict, out: Path) -> None:
    """Wall-time vs input tokens per cell, colored by agent."""
    cells = index_cells(summary)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, tier in zip(axes, TIERS):
        for a in AGENTS_SHORT:
            xs, ys = [], []
            for task, _, _ in CELLS:
                if f"/{tier}" not in _:
                    continue
            for task, tlabel in TASKS:
                c = cells.get((task, tier, a), {})
                for r in c.get("per_repeat") or []:
                    if r.get("prompt_tokens") and r.get("agent_wall_s"):
                        xs.append(r["prompt_tokens"])
                        ys.append(r["agent_wall_s"])
            if xs:
                ax.scatter(xs, ys, s=60, color=AGENT_COLOR[a], alpha=0.7,
                           edgecolors="white", linewidths=0.8, label=AGENT_LABEL[a])
        ax.set_title(f"{tier} tier  —  wall time vs prompt tokens")
        ax.set_xlabel("prompt tokens")
        ax.set_ylabel("agent wall time (s)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        if tier == "lite":
            ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    run_root = Path(sys.argv[1] if len(sys.argv) > 1
                    else str(IE_DIR.parent / "runs" / "matrix_v5_full")).resolve()
    # Fallback to committed curated snapshot if raw runs/ tree is absent
    if not run_root.is_dir():
        fallback = IE_DIR / "results" / run_root.name
        if fallback.is_dir():
            print(f"[plot_v5_summary] runs/{run_root.name} missing — using "
                  f"committed snapshot at {fallback}")
            run_root = fallback
    summary_path = run_root / "summary.json"
    if not summary_path.is_file():
        print(f"ERROR: {summary_path} missing — run aggregate_v5.py first")
        return 2
    summary = json.load(open(summary_path))

    out_dir = IE_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = run_root.name

    plot_pass_rate_heatmap(summary, out_dir / f"{tag}_pass_rate.png")
    print(f"wrote {out_dir}/{tag}_pass_rate.png")

    for task, tlabel in TASKS:
        plot_psnr_ssim_scatter(summary, task, tlabel,
                               out_dir / f"{tag}_scatter_{task}.png")
        print(f"wrote {out_dir}/{tag}_scatter_{task}.png")

    plot_efficiency(summary, out_dir / f"{tag}_efficiency.png")
    print(f"wrote {out_dir}/{tag}_efficiency.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
