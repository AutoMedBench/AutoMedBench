#!/usr/bin/env python3
"""Pro-tier summary plots in Claude-like cozy minimalism style.

Generates 5 charts from a detail_report.json:
  1. summary_metrics.png  — Performance overview (horizontal bars)
  2. summary_time.png     — Time breakdown by step (stacked horizontal)
  3. summary_api.png      — API usage (tokens per step)
  4. summary_cost.png     — Cost breakdown (donut chart)
  5. summary_failures.png — Error analysis heatmap
"""

import os


def _lazy_imports():
    """Lazy-load matplotlib to avoid import errors when not installed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    return plt, mpatches, np


# -------------------------------------------------------------------
# Claude-like cozy minimalism palette
# -------------------------------------------------------------------
STYLE = {
    "bg":         "#F5F0E8",
    "text":       "#3D3929",
    "text_light": "#8C8578",
    "accent":     "#D4775B",   # terracotta
    "blue":       "#6B8DB2",
    "sage":       "#8FA88A",
    "clay":       "#C4A882",
    "cream":      "#EDE6D8",
    "error":      "#C25D4E",
    "grid":       "#D6CFC2",
    "steps": ["#D4775B", "#6B8DB2", "#8FA88A", "#C4A882", "#A68BBF"],
}

STEP_LABELS = ["S1 Plan", "S2 Setup", "S3 Validate", "S4 Inference", "S5 Submit"]
STEP_KEYS = ["s1", "s2", "s3", "s4", "s5"]


def _apply_style(fig, ax):
    """Apply cozy minimalism to a figure/axes pair."""
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(STYLE["grid"])
    ax.spines["bottom"].set_color(STYLE["grid"])
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(colors=STYLE["text_light"], labelsize=9)


def _save(fig, path):
    """Save with tight layout and cozy margins."""
    plt, _, _ = _lazy_imports()
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=150, facecolor=STYLE["bg"],
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


# -------------------------------------------------------------------
# 1. Performance overview — horizontal bar chart
# -------------------------------------------------------------------

def plot_metrics(report: dict, path: str):
    """Horizontal bars for Organ Dice and Lesion Dice."""
    plt, mpatches, np = _lazy_imports()
    dm = report.get("diagnostic_metrics", {})
    metrics = {
        "Organ Dice":  dm.get("organ_dice", 0) or 0,
        "Lesion Dice": dm.get("lesion_dice", 0) or 0,
    }
    thresholds = {"Lesion Dice": 0.3}

    names = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=STYLE["bg"])
    _apply_style(fig, ax)

    bars = ax.barh(names, values, color=STYLE["accent"], height=0.55,
                   edgecolor="none", zorder=3)

    # Annotate values at bar ends
    for bar, v in zip(bars, values):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=10, color=STYLE["text"],
                fontweight="medium")

    # Threshold lines
    for name, thresh in thresholds.items():
        idx = names.index(name)
        y = idx
        ax.plot([thresh, thresh], [y - 0.35, y + 0.35],
                color=STYLE["text_light"], linewidth=1, linestyle="--",
                zorder=4)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("")
    ax.invert_yaxis()

    # Result tier badge
    tier_info = report.get("agentic_tier", {})
    medal = tier_info.get("medal_name", "N/A")
    rating = tier_info.get("rating", "?")
    ax.text(0.98, 0.02, f"{rating}  {medal}",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=STYLE["accent"], ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=STYLE["cream"],
                      edgecolor=STYLE["grid"], linewidth=0.5))

    ax.set_title("Performance Overview", fontsize=13, fontweight="medium",
                 color=STYLE["text"], pad=12)
    _save(fig, path)


# -------------------------------------------------------------------
# 2. Time breakdown — stacked horizontal bar
# -------------------------------------------------------------------

def plot_time(report: dict, path: str):
    """Stacked horizontal bar showing wall time per step."""
    plt, mpatches, np = _lazy_imports()
    tc = report.get("tool_calls", {})
    ps = tc.get("phase_summary", {})
    wall = report.get("runtime", {}).get("wall_time_s", 0)

    times = []
    for i, phase in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        s = ps.get(phase, {})
        times.append(s.get("total_exec_s", 0))

    # If no phase_summary, show total wall time as single bar
    if sum(times) == 0:
        times = [wall / 5] * 5  # distribute evenly as placeholder

    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor=STYLE["bg"])
    _apply_style(fig, ax)

    left = 0
    for i, (t, label) in enumerate(zip(times, STEP_LABELS)):
        bar = ax.barh(0, t, left=left, color=STYLE["steps"][i],
                      height=0.5, edgecolor="none")
        if t > wall * 0.08:  # label inline if segment is wide enough
            ax.text(left + t / 2, 0, f"{label}\n{t:.0f}s",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="medium")
        left += t

    # Total annotation
    ax.text(left + wall * 0.02, 0, f"Total: {wall:.0f}s",
            va="center", fontsize=10, color=STYLE["text"], fontweight="medium")

    ax.set_xlim(0, left * 1.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=STYLE["steps"][i], label=STEP_LABELS[i])
               for i in range(5)]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              frameon=False, ncol=5, labelcolor=STYLE["text_light"])

    ax.set_title("Time Breakdown", fontsize=13, fontweight="medium",
                 color=STYLE["text"], pad=12)
    _save(fig, path)


# -------------------------------------------------------------------
# 3. API usage — grouped bars for input/output tokens
# -------------------------------------------------------------------

def plot_api(report: dict, path: str):
    """Grouped bars: input tokens (blue) and output tokens (terracotta) per step."""
    plt, mpatches, np = _lazy_imports()
    rt = report.get("runtime", {})
    tc = report.get("tool_calls", {})
    ps = tc.get("phase_summary", {})

    # If phase_level token data exists, use it; else show totals
    in_tokens = []
    out_tokens = []
    calls = []
    for phase in ["S1", "S2", "S3", "S4", "S5"]:
        s = ps.get(phase, {})
        in_tokens.append(s.get("input_tokens", 0))
        out_tokens.append(s.get("output_tokens", 0))
        calls.append(s.get("calls", 0))

    # If no phase data, distribute totals as placeholder
    if sum(in_tokens) == 0:
        total_in = rt.get("input_tokens", 0)
        total_out = rt.get("output_tokens", 0)
        total_calls = rt.get("api_calls", 0)
        in_tokens = [total_in / 5] * 5
        out_tokens = [total_out / 5] * 5
        calls = [total_calls / 5] * 5

    x = np.arange(5)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=STYLE["bg"])
    _apply_style(fig, ax)

    bars1 = ax.bar(x - width / 2, [t / 1000 for t in in_tokens], width,
                   label="Input (K)", color=STYLE["blue"], edgecolor="none")
    bars2 = ax.bar(x + width / 2, [t / 1000 for t in out_tokens], width,
                   label="Output (K)", color=STYLE["accent"], edgecolor="none")

    # API call counts above bars
    for i, c in enumerate(calls):
        top = max(in_tokens[i], out_tokens[i]) / 1000
        if c > 0:
            ax.text(i, top + max(in_tokens) / 1000 * 0.05, f"{int(c)} calls",
                    ha="center", fontsize=7, color=STYLE["text_light"])

    ax.set_xticks(x)
    ax.set_xticklabels(STEP_LABELS, fontsize=9)
    ax.set_ylabel("Tokens (K)", fontsize=9, color=STYLE["text_light"])
    ax.legend(fontsize=8, frameon=False, labelcolor=STYLE["text_light"])

    # Subtitle with totals
    total_tok = rt.get("total_tokens", 0)
    total_api = rt.get("api_calls", 0)
    ax.text(0.5, -0.18, f"Total: {total_tok:,} tokens / {total_api} API calls",
            transform=ax.transAxes, fontsize=9, color=STYLE["text_light"],
            ha="center")

    ax.set_title("API Usage", fontsize=13, fontweight="medium",
                 color=STYLE["text"], pad=12)
    _save(fig, path)


# -------------------------------------------------------------------
# 4. Cost breakdown — donut chart
# -------------------------------------------------------------------

def plot_cost(report: dict, path: str):
    """Donut chart with S1-S5 cost segments, total in center."""
    plt, mpatches, np = _lazy_imports()
    rt = report.get("runtime", {})
    tc = report.get("tool_calls", {})
    ps = tc.get("phase_summary", {})
    total_cost = rt.get("estimated_cost_usd", 0)

    costs = []
    for phase in ["S1", "S2", "S3", "S4", "S5"]:
        s = ps.get(phase, {})
        costs.append(s.get("cost_usd", 0))

    # If no phase cost data, distribute total evenly
    if sum(costs) == 0 and total_cost > 0:
        costs = [total_cost / 5] * 5

    fig, ax = plt.subplots(figsize=(5, 5), facecolor=STYLE["bg"])
    fig.patch.set_facecolor(STYLE["bg"])

    if sum(costs) > 0:
        wedges, texts = ax.pie(
            costs, labels=None, colors=STYLE["steps"],
            startangle=90, pctdistance=0.85,
            wedgeprops=dict(width=0.3, edgecolor=STYLE["bg"], linewidth=2),
        )
    else:
        # No data — show empty ring
        ax.pie([1], colors=[STYLE["cream"]],
               wedgeprops=dict(width=0.3, edgecolor=STYLE["bg"], linewidth=2))

    # Center text
    ax.text(0, 0, f"${total_cost:.2f}", ha="center", va="center",
            fontsize=22, fontweight="bold", color=STYLE["text"])
    ax.text(0, -0.12, "total", ha="center", va="center",
            fontsize=10, color=STYLE["text_light"])

    # Legend
    legend_labels = [f"{STEP_LABELS[i]}  ${c:.2f}" for i, c in enumerate(costs)]
    patches = [mpatches.Patch(color=STYLE["steps"][i], label=legend_labels[i])
               for i in range(5)]
    ax.legend(handles=patches, loc="lower center", fontsize=8,
              frameon=False, ncol=3, labelcolor=STYLE["text_light"],
              bbox_to_anchor=(0.5, -0.08))

    ax.set_title("Cost Breakdown", fontsize=13, fontweight="medium",
                 color=STYLE["text"], pad=12)
    _save(fig, path)


# -------------------------------------------------------------------
# 5. Failure analysis — heatmap grid
# -------------------------------------------------------------------

def plot_failures(report: dict, path: str):
    """Grid heatmap: 5 steps x 5 error codes, colored by count."""
    plt, mpatches, np = _lazy_imports()
    ea = report.get("error_analysis", {})
    sf = report.get("step_failures", {})
    total_errors = ea.get("total_errors", 0)

    error_codes = ["E1", "E2", "E3", "E4", "E5"]
    steps = ["S1", "S2", "S3", "S4", "S5"]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=STYLE["bg"])
    _apply_style(fig, ax)

    if total_errors == 0:
        # No errors — show clean card
        ax.text(0.5, 0.5, "No errors detected", ha="center", va="center",
                fontsize=16, fontweight="medium", color=STYLE["sage"],
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title("Error Analysis", fontsize=13, fontweight="medium",
                     color=STYLE["text"], pad=12)
        _save(fig, path)
        return

    # Build 5x5 grid from step_failures
    grid = np.zeros((5, 5))
    for i, s_key in enumerate(["s1", "s2", "s3", "s4", "s5"]):
        code = sf.get(s_key)
        if code and code in error_codes:
            j = error_codes.index(code)
            grid[i][j] = 1

    # Create heatmap (import inside function — matplotlib is lazy-loaded)
    from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
    cmap = LinearSegmentedColormap.from_list(
        "cozy", [STYLE["cream"], STYLE["clay"], STYLE["accent"]]
    )
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(5))
    ax.set_xticklabels(error_codes, fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels(STEP_LABELS, fontsize=9)

    # Annotate cells
    for i in range(5):
        for j in range(5):
            val = int(grid[i][j])
            color = "white" if val > 0 else STYLE["text_light"]
            ax.text(j, i, str(val) if val > 0 else "",
                    ha="center", va="center", fontsize=11,
                    color=color, fontweight="bold" if val > 0 else "normal")

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Primary failure annotation
    primary = sf.get("primary_failure", "")
    if primary:
        ax.text(0.5, -0.15, f"Primary failure: {primary}",
                transform=ax.transAxes, fontsize=10, color=STYLE["error"],
                ha="center", fontweight="medium")

    ax.set_title("Error Analysis", fontsize=13, fontweight="medium",
                 color=STYLE["text"], pad=12)
    _save(fig, path)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def generate_summary_plots(detail_report: dict, output_dir: str) -> list:
    """Generate all Pro-tier summary plots. Returns list of saved paths."""
    os.makedirs(output_dir, exist_ok=True)

    plots = [
        ("summary_metrics.png", plot_metrics),
        ("summary_time.png", plot_time),
        ("summary_api.png", plot_api),
        ("summary_cost.png", plot_cost),
        ("summary_failures.png", plot_failures),
    ]

    saved = []
    for filename, plot_fn in plots:
        path = os.path.join(output_dir, filename)
        try:
            plot_fn(detail_report, path)
            saved.append(path)
        except Exception as e:
            print(f"[SummaryPlots] Warning: {filename} failed: {e}")

    return saved


# -------------------------------------------------------------------
# CLI: generate plots from a detail_report.json
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate Pro-tier summary plots")
    parser.add_argument("report_json", help="Path to detail_report.json")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as report)")
    args = parser.parse_args()

    with open(args.report_json) as f:
        report = json.load(f)

    out_dir = args.output_dir or os.path.dirname(args.report_json)
    saved = generate_summary_plots(report, out_dir)
    print(f"Generated {len(saved)} plots in {out_dir}")
    for p in saved:
        print(f"  {p}")
