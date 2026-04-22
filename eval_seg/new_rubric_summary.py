#!/usr/bin/env python3
"""Build new-rubric leaderboard + before/after comparison for kidney-lite.

Reads detail_report.json (old) and detail_report_new-rubrics.json (new) in
each run dir under runs/kuma/bench-*-kidney-lite, aggregates per-agent, and
prints a markdown summary.
"""
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/JHU/kuma_workspace/MedAgentsBench/eval_seg/runs/kuma')
KEEP = {'opus4.6', 'gemini3.1pro', 'glm5', 'gpt-5.4',
        'kimik2.5', 'minimax2.5', 'qwen3.5'}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_pair(run: Path):
    old = new = None
    old_p = run / "detail_report.json"
    new_p = run / "detail_report_new-rubrics.json"
    if old_p.exists():
        old = json.load(open(old_p))
    if new_p.exists():
        new = json.load(open(new_p))
    return old, new


def extract(report):
    if not report:
        return None
    ag = (report.get("agentic_score") or {})
    cl = (report.get("clinical_score") or {})
    tr = (report.get("agentic_tier") or {})
    ss = ag.get("step_scores", {}) or {}
    overall = f(tr.get("overall_score"))
    if overall is None:
        a, c = f(ag.get("score")), f(cl.get("score"))
        if a is not None and c is not None:
            overall = round(0.5 * a + 0.5 * c, 4)
    return {
        "agentic": f(ag.get("score")),
        "clinical": f(cl.get("score")),
        "overall": overall,
        "rating": tr.get("rating"),
        "medal": tr.get("medal_name"),
        "s1": f(ss.get("s1")),
        "s2": f(ss.get("s2")),
        "s3": f(ss.get("s3")),
        "s4": f(ss.get("s4")),
        "s5": f(ss.get("s5")),
        "organ_dice": f((report.get("diagnostic_metrics") or {}).get("organ_dice")),
        "lesion_dice": f((report.get("diagnostic_metrics") or {}).get("lesion_dice")),
    }


def mean(xs):
    xs = [v for v in xs if v is not None]
    return statistics.mean(xs) if xs else None


def collect():
    per_agent = defaultdict(list)
    for d in sorted(BASE.iterdir()):
        if not d.is_dir() or not d.name.endswith("-kidney-lite"):
            continue
        agent = d.name[len("bench-"):-len("-kidney-lite")]
        if agent not in KEEP:
            continue
        for run in d.iterdir():
            if not run.is_dir():
                continue
            old, new = load_pair(run)
            if old is None:
                continue
            rec = {"run": run.name, "old": extract(old), "new": extract(new)}
            per_agent[agent].append(rec)
    return per_agent


def print_report(per_agent, out_path=None):
    lines = []
    L = lines.append

    L("# Kidney-lite re-judge — new rubric summary\n")
    L("Clinical formula: `0.50*organ + 0.50*lesion`  (was `0.25*organ + 0.75*lesion`)")
    L("Step rubric: S1 = avg of 6 sub-criteria, S2 = avg of 5 sub-criteria, S3 = discrete 0/0.5/1, S4/S5 deterministic.")
    L("Weights: s1:0.25 s2:0.15 s3:0.35 s4:0.15 s5:0.10\n")

    # Per-agent averages
    L("## Per-agent averages (N = runs with new-rubric report)\n")
    L("| Agent | N | avg S1 | avg S2 | avg S3 | avg agentic | avg clinical | **avg overall** | Δoverall |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for agent, rs in per_agent.items():
        rs_new = [r for r in rs if r["new"] is not None]
        if not rs_new:
            continue
        s1 = mean(r["new"]["s1"] for r in rs_new)
        s2 = mean(r["new"]["s2"] for r in rs_new)
        s3 = mean(r["new"]["s3"] for r in rs_new)
        ag = mean(r["new"]["agentic"] for r in rs_new)
        cl = mean(r["new"]["clinical"] for r in rs_new)
        ov = mean(r["new"]["overall"] for r in rs_new)
        ov_old = mean(r["old"]["overall"] for r in rs_new if r["old"]["overall"] is not None)
        d = (ov - ov_old) if (ov is not None and ov_old is not None) else None
        rows.append((agent, len(rs_new), s1, s2, s3, ag, cl, ov, d))
    rows.sort(key=lambda r: -(r[7] or 0))
    for agent, n, s1, s2, s3, ag, cl, ov, d in rows:
        fmt = lambda v, p=4: f"{v:.{p}f}" if v is not None else "—"
        dfmt = f"{d:+.4f}" if d is not None else "—"
        L(f"| {agent} | {n} | {fmt(s1,2)} | {fmt(s2,2)} | {fmt(s3,2)} | {fmt(ag)} | {fmt(cl)} | **{fmt(ov)}** | {dfmt} |")

    # Before/after overall score comparison
    L("\n## Before/after overall score (mean)\n")
    L("| Agent | old overall | new overall | Δ |")
    L("|---|---:|---:|---:|")
    for agent, n, s1, s2, s3, ag, cl, ov, d in rows:
        rs_new = [r for r in per_agent[agent] if r["new"] is not None]
        ov_old = mean(r["old"]["overall"] for r in rs_new if r["old"]["overall"] is not None)
        L(f"| {agent} | {ov_old:.4f} | {ov:.4f} | {(ov - ov_old):+.4f} |")

    # Rating distribution
    L("\n## Rating distribution (new rubric)\n")
    L("| Agent | A | B | C | F |")
    L("|---|---:|---:|---:|---:|")
    for agent, n, *_ in rows:
        rs_new = [r for r in per_agent[agent] if r["new"] is not None]
        c = Counter(r["new"]["rating"] for r in rs_new)
        L(f"| {agent} | {c.get('A',0)} | {c.get('B',0)} | {c.get('C',0)} | {c.get('F',0)} |")

    # S1 sub-score shift (avg)
    L("\n## S1 step score shift (old → new)\n")
    L("| Agent | old S1 avg | new S1 avg | Δ |")
    L("|---|---:|---:|---:|")
    for agent, n, *_ in rows:
        rs_new = [r for r in per_agent[agent] if r["new"] is not None]
        os1 = mean(r["old"]["s1"] for r in rs_new if r["old"]["s1"] is not None)
        ns1 = mean(r["new"]["s1"] for r in rs_new if r["new"]["s1"] is not None)
        if os1 is not None and ns1 is not None:
            L(f"| {agent} | {os1:.3f} | {ns1:.3f} | {(ns1-os1):+.3f} |")

    # S2 shift
    L("\n## S2 step score shift (old → new)\n")
    L("| Agent | old S2 avg | new S2 avg | Δ |")
    L("|---|---:|---:|---:|")
    for agent, n, *_ in rows:
        rs_new = [r for r in per_agent[agent] if r["new"] is not None]
        os2 = mean(r["old"]["s2"] for r in rs_new if r["old"]["s2"] is not None)
        ns2 = mean(r["new"]["s2"] for r in rs_new if r["new"]["s2"] is not None)
        if os2 is not None and ns2 is not None:
            L(f"| {agent} | {os2:.3f} | {ns2:.3f} | {(ns2-os2):+.3f} |")

    # S3 shift
    L("\n## S3 step score shift (old → new)\n")
    L("| Agent | old S3 avg | new S3 avg | Δ |")
    L("|---|---:|---:|---:|")
    for agent, n, *_ in rows:
        rs_new = [r for r in per_agent[agent] if r["new"] is not None]
        os3 = mean(r["old"]["s3"] for r in rs_new if r["old"]["s3"] is not None)
        ns3 = mean(r["new"]["s3"] for r in rs_new if r["new"]["s3"] is not None)
        if os3 is not None and ns3 is not None:
            L(f"| {agent} | {os3:.3f} | {ns3:.3f} | {(ns3-os3):+.3f} |")

    text = "\n".join(lines)
    print(text)
    if out_path:
        Path(out_path).write_text(text)
        print(f"\n[saved to {out_path}]")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else None
    print_report(collect(), out)
