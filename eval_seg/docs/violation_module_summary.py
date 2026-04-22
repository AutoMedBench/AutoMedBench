#!/usr/bin/env python3
"""Violation Analysis Module — single-page proposal diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Tokyo Night palette ────────────────────────────────────────────────
BG      = "#1A1B26"
CARD    = "#24283B"
CARD2   = "#1F2335"
BORDER  = "#3B4261"
TEXT    = "#C0CAF5"
DIM     = "#565F89"
WHITE   = "#E0E0E0"
GREEN   = "#9ECE6A"
GREEN_B = "#1A2E1A"
RED     = "#F7768E"
RED_B   = "#2D1B2E"
YELLOW  = "#E0AF68"
BLUE    = "#7AA2F7"
BLUE_B  = "#1A2040"
PURPLE  = "#BB9AF7"
PURPLE_B= "#251B35"
ORANGE  = "#FF9E64"
ORANGE_B= "#2D2218"
CYAN    = "#7DCFFF"
TEAL    = "#73DACA"
GRAY    = "#444B6A"

DOCS = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/JHU/kuma_workspace/MedAgentsBench/eval_seg/docs"

def box(ax, x, y, w, h, fc, ec, text="", fs=10, tc=None, fw="normal", rad=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        fc=fc, ec=ec, lw=1.2, zorder=2))
    if text:
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fs, color=tc or TEXT, fontweight=fw, zorder=3,
                linespacing=1.35)


fig, ax = plt.subplots(figsize=(14, 9.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ══════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════
ax.text(0.50, 0.975, "Violation Analysis Module",
        ha="center", va="top", fontsize=21, color=WHITE, fontweight="bold")
ax.text(0.50, 0.950, "detect  ·  kill  ·  zero  ·  report",
        ha="center", va="top", fontsize=10, color=DIM)

# ══════════════════════════════════════════════════════════════════════
# TOP — Agent workflow S1→S5 with kill marker
# ══════════════════════════════════════════════════════════════════════
steps = ["S1 Research", "S2 Setup", "S3 Validate", "S4 Inference", "S5 Submit"]
sx, sy, sw, sh, sg = 0.04, 0.865, 0.155, 0.048, 0.03

for i, s in enumerate(steps):
    x = sx + i*(sw+sg)
    alive = i < 2
    box(ax, x, sy, sw, sh,
        GREEN_B if alive else CARD2,
        GREEN if alive else GRAY,
        s, fs=9.5,
        tc=GREEN if alive else DIM, fw="bold")
    if i < 4:
        cx = x + sw + sg/2
        ax.text(cx, sy+sh/2, "→", ha="center", va="center", fontsize=9,
                color=GREEN if i < 1 else GRAY)

# kill arrow
kx = sx + 1*(sw+sg) + sw*0.78
ax.annotate("", xy=(kx, sy+sh), xytext=(kx, sy+sh+0.032),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
ax.text(kx, sy+sh+0.038, "KILL", ha="center", va="bottom",
        fontsize=9, color=RED, fontweight="bold")

# X on S3-S5
for i in [2, 3, 4]:
    cx = sx + i*(sw+sg) + sw/2
    cy = sy + sh/2
    r = 0.015
    ax.plot([cx-r, cx+r], [cy-r, cy+r], color=RED, lw=2.5, alpha=0.55)
    ax.plot([cx-r, cx+r], [cy+r, cy-r], color=RED, lw=2.5, alpha=0.55)

# ══════════════════════════════════════════════════════════════════════
# MIDDLE-LEFT — Flow pipeline (compact horizontal)
# ══════════════════════════════════════════════════════════════════════
ax.text(0.04, 0.81, "Flow", fontsize=9, color=DIM, fontweight="bold")

flow = [
    ("execute\ncode",     CARD,     BLUE,   BLUE),
    ("4-layer\ncheck",    PURPLE_B, PURPLE, PURPLE),
    ("violation?",        RED_B,    RED,    RED),
    ("zero from\nstep N→", ORANGE_B, ORANGE, ORANGE),
    ("report\nrating=F",  CARD,     TEAL,   TEAL),
]
fx, fy, fw_, fh, fg = 0.04, 0.725, 0.145, 0.06, 0.03

for i, (t, fc, ec, tc) in enumerate(flow):
    x = fx + i*(fw_+fg)
    box(ax, x, fy, fw_, fh, fc, ec, t, fs=8.5, tc=tc, fw="bold")
    if i < len(flow)-1:
        ax.annotate("", xy=(x+fw_+fg, fy+fh/2), xytext=(x+fw_, fy+fh/2),
                    arrowprops=dict(arrowstyle="-|>", color=DIM, lw=1))

# ══════════════════════════════════════════════════════════════════════
# MIDDLE — Penalty matrix (compact)
# ══════════════════════════════════════════════════════════════════════
ax.text(0.04, 0.685, "Penalty Matrix", fontsize=9, color=DIM, fontweight="bold")

headers = ["S1", "S2", "S3", "S4", "S5", "Clin.", "Rating"]
rows = [
    ("Normal",  [.85,.70,.60,.80,.75], .65, "A", None),
    ("Kill@S1", [0,0,0,0,0],           0,   "F", 0),
    ("Kill@S2", [.85,0,0,0,0],         0,   "F", 1),
    ("Kill@S3", [.85,.70,0,0,0],       0,   "F", 2),
    ("Kill@S4", [.85,.70,.60,0,0],     0,   "F", 3),
    ("Kill@S5", [.85,.70,.60,.80,0],   0,   "F", 4),
]

mx, my = 0.135, 0.635
cw, ch = 0.088, 0.042
hh = 0.030

# headers
for j, h in enumerate(headers):
    x = mx + j*cw
    hc = BLUE if j < 5 else ORANGE if j == 5 else PURPLE
    box(ax, x+.002, my+.002, cw-.004, hh, CARD, hc, h,
        fs=8, tc=hc, fw="bold", rad=.006)

# rows
for i, (label, scores, clin, rating, kill) in enumerate(rows):
    y = my - (i+1)*ch
    lc = GREEN if kill is None else RED
    ax.text(mx-.008, y+ch/2, label, ha="right", va="center",
            fontsize=7.5, color=lc, fontweight="bold")

    vals = scores + [clin, rating]
    for j, v in enumerate(vals):
        x = mx + j*cw
        if j == 6:
            fc = GREEN_B if v!="F" else RED_B
            ec = GREEN if v!="F" else RED
            tc = GREEN if v!="F" else RED
            box(ax, x+.002, y+.002, cw-.004, ch-.004, fc, ec,
                str(v), fs=11, tc=tc, fw="bold", rad=.005)
        else:
            if kill is None:
                fc, ec, tc = GREEN_B, GREEN, GREEN
            elif j < kill:
                fc, ec, tc = CARD2, BORDER, DIM
            else:
                fc, ec, tc = RED_B, RED, RED
            t = f"{v:.1f}" if isinstance(v, float) and v > 0 else "0"
            box(ax, x+.002, y+.002, cw-.004, ch-.004, fc, ec,
                t, fs=8.5, tc=tc, fw="bold", rad=.005)

# ══════════════════════════════════════════════════════════════════════
# BOTTOM-LEFT — Bar chart (kill@S2 example)
# ══════════════════════════════════════════════════════════════════════
ax.text(0.04, 0.355, "Score Impact — Kill at S2", fontsize=9, color=DIM, fontweight="bold")

labels = ["S1", "S2", "S3", "S4", "S5", "Clin"]
baseline = [.85, .70, .60, .80, .75, .65]
after    = [.85,  0,   0,   0,   0,   0 ]
kill_at  = 1

# mini axes via inset
ax_bar = fig.add_axes([0.055, 0.06, 0.40, 0.28])
ax_bar.set_facecolor(CARD2)
for spine in ax_bar.spines.values():
    spine.set_color(BORDER)
ax_bar.set_ylim(0, 1.08)
ax_bar.tick_params(colors=DIM, labelsize=7)

x = np.arange(len(labels))
bw = 0.30

# baseline ghost bars
ax_bar.bar(x - bw/2 - 0.02, baseline, bw,
           color=GRAY, alpha=0.18, edgecolor=GRAY, lw=0.4)

# after bars
colors = []
for i, v in enumerate(after):
    if i < kill_at:    colors.append(DIM)
    elif v > 0:        colors.append(GREEN)
    else:              colors.append(RED)

for i, (v, c) in enumerate(zip(after, colors)):
    h = max(v, 0.02)
    ax_bar.bar(i + bw/2 + 0.02, h, bw,
               color=c, alpha=0.35, edgecolor=c, lw=1)
    lbl = f"{v:.2f}" if v > 0 else "0"
    ax_bar.text(i + bw/2 + 0.02, v + 0.04, lbl,
                ha="center", va="bottom", fontsize=7, color=c, fontweight="bold")

ax_bar.axvline(kill_at - 0.5, color=RED, lw=1.2, ls="--", alpha=0.5)
ax_bar.text(kill_at - 0.5, 1.04, "kill", ha="center", fontsize=7,
            color=RED, fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, color=TEXT, fontsize=8)
ax_bar.text(len(labels)-1, 0.95, "gray=baseline", fontsize=6.5,
            color=GRAY, ha="right")

# ══════════════════════════════════════════════════════════════════════
# BOTTOM-RIGHT — Agent-facing messages (compact)
# ══════════════════════════════════════════════════════════════════════
# System prompt card
px, py, pw, ph = 0.52, 0.185, 0.45, 0.16
box(ax, px, py, pw, ph, CARD, YELLOW, rad=.01)
ax.text(px+.01, py+ph-.01, "System Prompt → Agent",
        fontsize=8.5, color=YELLOW, fontweight="bold", ha="left", va="top")
prompt = (
    "ALLOWED  {data_dir}/, {output_dir}/, pip, downloads\n"
    "BLOCKED  find /, ls /, os.walk('/'), ../../,\n"
    "         /proc, harness code, ground_truth\n"
    "PENALTY  kill → remaining S-scores=0,\n"
    "         clinical=0, rating=F"
)
ax.text(px+.01, py+ph-.035, prompt, fontsize=7, color=TEXT,
        fontfamily="monospace", ha="left", va="top", linespacing=1.4)

# Kill message card
kx, ky, kw, kh = 0.52, 0.01, 0.45, 0.16
box(ax, kx, ky, kw, kh, RED_B, RED, rad=.01)
ax.text(kx+.01, ky+kh-.01, "Kill Message → Agent",
        fontsize=8.5, color=RED, fontweight="bold", ha="left", va="top")
kill = (
    'VIOLATION at turn 8: "ls /lustre"\n'
    "execution terminated\n"
    "S2..S5 → 0, clinical → 0\n"
    "rating → F, partial outputs discarded"
)
ax.text(kx+.01, ky+kh-.035, kill, fontsize=7, color=TEXT,
        fontfamily="monospace", ha="left", va="top", linespacing=1.4)

fig.savefig(f"{DOCS}/violation_module.png",
            dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.25)
plt.close()
print(f"Saved: {DOCS}/violation_module.png")
