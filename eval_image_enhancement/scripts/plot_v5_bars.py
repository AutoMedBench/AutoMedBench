#!/usr/bin/env python3
"""v5 bar charts (v3-style): pass-rate + S1-S5 stage timing + token fairness.

Reads runs/matrix_v5*/<task>/<tier>/<agent>_r<i>/detail_report.json and
aggregates across the 3 repeats.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

IE_DIR = Path(__file__).resolve().parent.parent
FIGS = IE_DIR / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

AGENTS = ["gemini31pro", "gpt54", "opus46", "glm5", "minimax25", "qwen35"]
AGENT_LABEL = {
    "gemini31pro": "Gemini 3.1 Pro", "gpt54": "GPT-5.4",
    "opus46":      "Claude Opus 4.6", "glm5": "GLM-5",
    "minimax25":   "MiniMax 2.5",     "qwen35": "Qwen 3.5",
}
TIERS = ["lite", "standard"]
TASKS = [("ldct-denoising-task", "LDCT Denoising (10p)"),
         ("mri-sr-task",         "MRI-SR ×2 (10p)")]
STAGES = ["S1", "S2", "S3", "S4", "S5"]
STAGE_COLOR = {"S1": "#4C9EEB", "S2": "#6ECE4D", "S3": "#F2A65A",
               "S4": "#E35B5B", "S5": "#AA78E0"}
STAGE_RE = {s: re.compile(
    rf"(?:^|\n)\s*(?:#{{1,6}}\s*\**\s*|\*\*\s*)?{s}(?:\*\*)?"
    rf"(?:\s*[:\-—]|\s+(?:PLAN|RESEARCH|SETUP|VALIDATE|INFERENCE|SUBMIT))",
    re.IGNORECASE | re.MULTILINE) for s in STAGES}


def detect_stage(content: str, prev: str) -> str:
    earliest_s, earliest_pos = None, 10 ** 9
    for s in STAGES:
        m = STAGE_RE[s].search(content or "")
        if m and m.start() < earliest_pos:
            earliest_pos, earliest_s = m.start(), s
    return earliest_s or prev


def load_one_repeat(rd: Path) -> dict | None:
    rp = rd / "detail_report.json"
    if not rp.is_file():
        return None
    r = json.load(open(rp))
    sc = r.get("scores") or {}
    fmt = r.get("format") or {}
    asum = r.get("agent_summary") or {}
    orch = r.get("orchestrator") or {}
    pr = r.get("pass_rate") or {}
    rating = (r.get("rating") or {}).get("rating", "F")

    msgs_p = rd / "workspace" / "process" / "messages.json"
    tlog_p = rd / "workspace" / "process" / "tool_log.jsonl"
    stage_exec = {k: 0.0 for k in STAGES}
    stage_api = {k: 0.0 for k in STAGES}
    turn_stage = {}

    if msgs_p.is_file():
        try:
            msgs = json.load(open(msgs_p))
            current = "S1"
            turn = 0
            for m in msgs:
                if m.get("role") == "assistant":
                    turn += 1
                    current = detect_stage(m.get("content") or "", current)
                    turn_stage[turn] = current
        except Exception:
            pass

    if tlog_p.is_file():
        with open(tlog_p) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                t = e.get("turn", 0)
                st = turn_stage.get(t, "S1")
                stage_exec[st] += float((e.get("result") or {}).get("elapsed_s") or 0)

    total_turns = asum.get("turns") or max(1, len(turn_stage))
    api_time = float(asum.get("api_time_s") or 0)
    if turn_stage:
        turns_per_stage = {s: sum(1 for v in turn_stage.values() if v == s)
                           for s in STAGES}
        for s in STAGES:
            stage_api[s] = api_time * turns_per_stage[s] / max(1, total_turns)

    return {
        "rating": rating,
        "pass_rate": pr.get("pass_rate"),
        "mean_psnr": sc.get("mean_psnr"),
        "mean_ssim": sc.get("mean_ssim"),
        "n_valid": fmt.get("n_valid", 0),
        "n_patients": fmt.get("n_patients", 0),
        "turns": asum.get("turns", 0),
        "wall": asum.get("elapsed_s") or orch.get("agent_wall_s", 0),
        "submit": asum.get("submitted", False),
        "in_tok": asum.get("prompt_tokens", 0),
        "out_tok": asum.get("completion_tokens", 0),
        "stage_exec": stage_exec,
        "stage_api": stage_api,
    }


def load_cell(root: Path, task: str, tier: str, agent: str) -> dict:
    reps = []
    for r in range(10):  # up to 10 repeats
        rd = root / task / tier / f"{agent}_r{r}"
        if not rd.is_dir():
            break
        d = load_one_repeat(rd)
        if d is not None:
            reps.append(d)
    if not reps:
        return {}

    def avg(k):
        vs = [r[k] for r in reps if r.get(k) is not None]
        return float(np.mean(vs)) if vs else None

    # Majority rating (use worst-case for visualization: most common letter across repeats)
    ratings = [r["rating"] for r in reps]
    # Weighted: if ≥2 repeats agree, use that; else use worst
    from collections import Counter
    cnt = Counter(ratings).most_common()
    rating = cnt[0][0] if cnt[0][1] >= 2 else max(ratings, key=lambda x: "FCBA".index(x))

    stage_exec = {s: float(np.mean([r["stage_exec"][s] for r in reps])) for s in STAGES}
    stage_api  = {s: float(np.mean([r["stage_api"][s]  for r in reps])) for s in STAGES}

    return {
        "n_repeats":  len(reps),
        "rating":     rating,
        "ratings_all": ratings,
        "pass_rate":  avg("pass_rate"),
        "psnr":       avg("mean_psnr"),
        "ssim":       avg("mean_ssim"),
        "wall":       avg("wall"),
        "turns":      avg("turns"),
        "in_tok":     avg("in_tok"),
        "out_tok":    avg("out_tok"),
        "n_valid":    avg("n_valid"),
        "n_total":    avg("n_patients"),
        "stage_exec": stage_exec,
        "stage_api":  stage_api,
    }


# ---------------------------------------------------------------------------
# Figure 1: pass-rate bar chart (v3 style)
# ---------------------------------------------------------------------------
def fig_pass_rate(root: Path, tag: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for ax, (task, tname) in zip(axes, TASKS):
        x = np.arange(len(AGENTS))
        w = 0.36
        for i, tier in enumerate(TIERS):
            rates, psnrs, ratings, valid = [], [], [], []
            for agent in AGENTS:
                info = load_cell(root, task, tier, agent)
                rates.append((info.get("pass_rate") or 0) * 100)
                psnrs.append(info.get("psnr"))
                ratings.append(info.get("rating", "F"))
                valid.append(info.get("n_valid") or 0)
            colors = []
            for r, v in zip(ratings, valid):
                if v == 0:
                    colors.append("#7f8c8d")
                elif r == "A":
                    colors.append("#2ECC71")
                elif r == "B":
                    colors.append("#3498DB")
                elif r == "C":
                    colors.append("#F2A65A")
                else:
                    colors.append("#E74C3C")
            offs = x + (i - 0.5) * w
            bars = ax.bar(offs, rates, w, color=colors, edgecolor="black",
                          linewidth=0.5, label=tier.capitalize())
            for bx, rate, psnr, rt, v in zip(bars, rates, psnrs, ratings, valid):
                if v == 0:
                    ax.text(bx.get_x() + bx.get_width() / 2, 3,
                            "F\n(no output)", ha="center", va="bottom",
                            fontsize=7.5, color="red", style="italic")
                    bx.set_hatch("////"); bx.set_alpha(0.45)
                else:
                    p_s = f"{psnr:.1f}" if psnr is not None else "—"
                    ax.text(bx.get_x() + bx.get_width() / 2, rate + 1.5,
                            f"{rate:.0f}%\n[{rt}]\n{p_s}dB",
                            ha="center", va="bottom", fontsize=7.2,
                            linespacing=1.1)
        for ti, tier in enumerate(TIERS):
            for i in range(len(AGENTS)):
                ax.text(i + (ti - 0.5) * w, -4, tier[0].upper(),
                        ha="center", va="top", fontsize=7.5, color="#555")
        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABEL[a] for a in AGENTS],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Per-case pass rate (%)\nSSIM ≥ classical − 0.02", fontsize=10)
        ax.set_ylim(-8, 115)
        ax.set_title(tname, fontsize=11, fontweight="bold")
        ax.grid(axis="y", ls=":", alpha=0.4)

    handles = [
        mpatches.Patch(color="#2ECC71", label="A (beats DNN baseline)"),
        mpatches.Patch(color="#3498DB", label="B (matches DNN baseline)"),
        mpatches.Patch(color="#F2A65A", label="C (matches classical)"),
        mpatches.Patch(color="#E74C3C", label="F (below classical)"),
        mpatches.Patch(color="#7f8c8d", label="F (no output / aborted)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    fig.suptitle(f"{tag} — per-case pass rate (mean over 3 repeats; "
                 f"L=Lite, S=Standard; bar color = modal rating)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    out = FIGS / f"{tag}_bars_pass_rate.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: S1-S5 stacked stage timing per cell
# ---------------------------------------------------------------------------
def fig_stage_timing(root: Path, tag: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for row, (task, tname) in enumerate(TASKS):
        for col, tier in enumerate(TIERS):
            ax = axes[row, col]
            x = np.arange(len(AGENTS))
            bottom = np.zeros(len(AGENTS))
            totals, turns, ratings, pass_rates = [], [], [], []
            for agent in AGENTS:
                info = load_cell(root, task, tier, agent)
                totals.append(info.get("wall") or 0)
                turns.append(info.get("turns") or 0)
                ratings.append(info.get("rating", "F"))
                pass_rates.append(info.get("pass_rate") or 0)
            for s in STAGES:
                stage_vals = []
                for agent in AGENTS:
                    info = load_cell(root, task, tier, agent)
                    e = (info.get("stage_exec") or {}).get(s, 0)
                    a = (info.get("stage_api")  or {}).get(s, 0)
                    stage_vals.append(e + a)
                ax.bar(x, stage_vals, bottom=bottom, width=0.62,
                       color=STAGE_COLOR[s], edgecolor="white", linewidth=0.5,
                       label=s if row == 0 and col == 0 else None)
                bottom = bottom + np.array(stage_vals)
            y_max = max(totals) if totals else 1
            for i in range(len(AGENTS)):
                ax.text(x[i], totals[i] + y_max * 0.03,
                        f"{totals[i]:.0f}s\n{turns[i]:.0f}t\n"
                        f"{pass_rates[i]*100:.0f}%[{ratings[i]}]",
                        ha="center", va="bottom", fontsize=7.5,
                        linespacing=1.1, fontweight="bold")
            ax.set_ylim(0, y_max * 1.25)
            ax.set_xticks(x)
            ax.set_xticklabels([AGENT_LABEL[a] for a in AGENTS],
                               rotation=25, ha="right", fontsize=8.5)
            ax.set_ylabel("Wall-clock per stage (s)\n[exec + allocated API]",
                          fontsize=10)
            ax.set_title(f"{tname}  —  {tier.upper()}",
                         fontsize=11, fontweight="bold")
            ax.grid(axis="y", ls=":", alpha=0.3)

    handles = [mpatches.Patch(color=STAGE_COLOR[s], label=s) for s in STAGES]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.97), fontsize=10, frameon=False,
               title="Stage (stacked bottom → top)")
    fig.suptitle(f"{tag} — S1-S5 wall-clock per agent × tier × task "
                 f"(mean over 3 repeats)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIGS / f"{tag}_bars_stage_timing.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: token fairness
# ---------------------------------------------------------------------------
def fig_token_fairness(root: Path, tag: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (task, tname) in zip(axes, TASKS):
        x = np.arange(len(AGENTS))
        w = 0.36
        for i, tier in enumerate(TIERS):
            tpc = []
            for agent in AGENTS:
                info = load_cell(root, task, tier, agent)
                in_tok = info.get("in_tok") or 0
                n = info.get("n_total") or 1
                tpc.append(in_tok / max(1, n))
            offs = x + (i - 0.5) * w
            bars = ax.bar(offs, tpc, w, label=tier.capitalize(),
                          color="#3498DB" if tier == "lite" else "#F2A65A",
                          edgecolor="black", linewidth=0.5)
            for bx, t in zip(bars, tpc):
                ax.text(bx.get_x() + bx.get_width() / 2,
                        max(t * 1.05, 1), f"{t:,.0f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABEL[a] for a in AGENTS],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Input tokens / case  (log scale)", fontsize=10)
        ax.set_title(tname, fontsize=11, fontweight="bold")
        ax.grid(axis="y", ls=":", alpha=0.4, which="both")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"{tag} — token fairness: input tokens per patient "
                 f"(Lite vs Standard)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGS / f"{tag}_bars_token_fairness.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else str(IE_DIR.parent / "runs" / "matrix_v5_full")).resolve()
    # Fallback to committed curated snapshot if the raw runs/ tree is absent
    if not root.is_dir():
        fallback = IE_DIR / "results" / root.name
        if fallback.is_dir():
            print(f"[plot_v5_bars] runs/{root.name} missing — using committed "
                  f"snapshot at {fallback} (stage-timing figure will be blank "
                  f"because per-run process/messages.json is not committed)")
            root = fallback
    tag = root.name
    fig_pass_rate(root, tag)
    fig_stage_timing(root, tag)
    fig_token_fairness(root, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
