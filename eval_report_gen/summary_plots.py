#!/usr/bin/env python3
"""Generate lightweight SVG summary plots without external dependencies."""

from __future__ import annotations

from pathlib import Path


def _svg_bar(title: str, items: list[tuple[str, float]], path: str | Path) -> str:
    path = Path(path)
    width = 640
    height = 120 + 40 * len(items)
    max_value = max([value for _, value in items] + [1.0])
    rows = []
    y = 60
    for label, value in items:
        bar_width = int((width - 220) * (value / max_value))
        rows.append(
            f'<text x="20" y="{y}" font-size="14" font-family="monospace">{label}</text>'
            f'<rect x="220" y="{y-14}" width="{bar_width}" height="18" fill="#28536b" />'
            f'<text x="{230 + bar_width}" y="{y}" font-size="14" font-family="monospace">{value:.3f}</text>'
        )
        y += 36
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="#f7f5ef" />'
        f'<text x="20" y="30" font-size="22" font-family="monospace" fill="#102a43">{title}</text>'
        + "".join(rows)
        + "</svg>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return str(path)


def generate_summary_plots(detail_report: dict, output_dir: str | Path) -> list[str]:
    output_dir = Path(output_dir)
    metrics = detail_report["diagnostic_metrics"]
    runtime = detail_report["runtime"]
    steps = detail_report["agentic_score"]["step_scores"]
    failure = detail_report.get("error_analysis", {}).get("code_counts", {})

    paths = [
        _svg_bar(
            "Clinical Metrics",
            [
                ("observation_f1", float(metrics.get("observation_f1", 0.0))),
                ("report_similarity", float(metrics.get("report_similarity", 0.0))),
                ("label_exact_match", float(metrics.get("label_exact_match", 0.0))),
            ],
            output_dir / "clinical_metrics.svg",
        ),
        _svg_bar(
            "Workflow Steps",
            [(key, float(value or 0.0)) for key, value in steps.items()],
            output_dir / "workflow_steps.svg",
        ),
        _svg_bar(
            "Runtime",
            [
                ("wall_time_s", float(runtime.get("wall_time_s", 0.0))),
                ("api_calls", float(runtime.get("api_calls", 0.0))),
                ("code_executions", float(runtime.get("code_executions", 0.0))),
            ],
            output_dir / "runtime.svg",
        ),
        _svg_bar(
            "Error Codes",
            [(key, float(value)) for key, value in sorted(failure.items())],
            output_dir / "errors.svg",
        ),
    ]
    return paths
