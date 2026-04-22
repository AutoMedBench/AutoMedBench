#!/usr/bin/env python3
"""LLM-as-judge for eval_report_gen.

Offline mode is heuristic and has no external dependencies.
Online mode uses an OpenAI-compatible endpoint if configured.
"""

from __future__ import annotations

import base64
import json
import re
import struct
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from api_key_loader import load_api_keys
except ModuleNotFoundError:  # pragma: no cover - import path depends on invocation style
    from eval_report_gen.api_key_loader import load_api_keys

try:
    from config_io import load_config
except ModuleNotFoundError:  # pragma: no cover - import path depends on invocation style
    from eval_report_gen.config_io import load_config


@dataclass
class JudgeVerdict:
    s1_plan_score: float = 0.0
    s1_rationale: str = ""
    s1_failure: str | None = None
    s2_setup_score: float = 0.0
    s2_rationale: str = ""
    s2_failure: str | None = None
    s3_validate_score: float = 0.0
    s3_rationale: str = ""
    s3_failure: str | None = None
    detected_failure: str | None = None
    failure_explanation: str = ""
    overall_rationale: str = ""
    judge_model: str = ""
    judge_backend: str = ""
    judge_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a case-based chest X-ray report-generation benchmark.

Score only the workflow stages S1-S3:
- S1 Plan: plan quality and task understanding
- S2 Setup: whether the agent established a workable generation pipeline
- S3 Validate: whether the agent validated on one or more pilot cases before full batch generation

Scoring rule:
- Each stage score is the mean of its binary checks.
- Score each applicable check as `1` if the evidence clearly satisfies it, else `0`.
- Standard/Pro-only checks are excluded in Lite.

Path convention:
- The judge receives absolute artifact paths.
- Every rubric path below is relative to `OUTPUT_DIR = <run_dir>/outputs`.
- `plan/plan.md` means `OUTPUT_DIR/plan/plan.md`.
- `plan/plan.png` means `OUTPUT_DIR/plan/plan.png`.
- `agent_outputs/<case_id>/report.txt` means `OUTPUT_DIR/agent_outputs/<case_id>/report.txt`.

Definitions:
- `case`: the benchmark input unit under `public/<case_id>/`. In the source dataset this corresponds to one study, but score the rubric in terms of `case`.
- `case images`: the one or more JPEG files listed in `public/<case_id>/manifest.json["image_files"]`.
- `selected inference entrypoint`: the exact repo-defined script, notebook, CLI, or `python -m ...` path chosen in `plan.md` for the selected method.
- `official export logic`: the selected method's own documented decode / post-process / section-extraction path that turns raw model output into the final report text. If the method has no documented export helper, treat this as conditional and look for an explicit note in `plan.md` before giving credit to a custom adapter.

S1 checks:
1. `plan/plan.md` exists.
2. `plan.md` names the selected model.
3. `plan.md` names the selected inference entrypoint.
4. `plan.md` reflects the benchmark unit correctly: one case is one full case input, not one JPEG.
5. `plan.md` states that all files in `public/<case_id>/manifest.json["image_files"]` must be used.
6. `plan.md` states the exact output path `agent_outputs/<case_id>/report.txt`.
7. `plan.md` requires pilot validation before full-batch generation.
8. `plan.md` defines at least one specific pilot check.
9. Standard/Pro only: `plan/plan.png` exists.
10. Standard/Pro only: `plan.png` shows the same selected model and the same stage order as `plan.md`.
11. Standard/Pro only: `plan.md` compares multiple candidate methods and justifies the selected method.

S2 checks:
1. Successful execution runs the selected inference pipeline.
2. Setup reads `public/<case_id>/manifest.json`.
3. Setup loads case images from `manifest.json["image_files"]`.
4. Setup preserves one case as one input unit by grouping all JPEGs listed in `manifest.json["image_files"]` for a single report, rather than treating each JPEG as its own case.
5. Setup writes `agent_outputs/<case_id>/report.txt`.
6. Setup uses the exact selected inference entrypoint named in `plan.md`.
7. If the selected method defines official export logic, setup uses that exact decode / post-process path when producing `report.txt`.

S3 checks:
1. At least one pilot case runs before full-batch generation.
2. The pilot run uses the same selected model, checkpoint, inference entrypoint, and export/decode path as S2.
3. The pilot run writes `agent_outputs/<case_id>/report.txt`.
4. The saved pilot `report.txt` is opened after the run.
5. Validation checks `manifest.json["image_files"]`.
6. Validation confirms that all JPEGs listed in `manifest.json["image_files"]` for that case were used.
7. If the first pilot run fails, a later pilot rerun occurs before full-batch generation.

Important:
- Inspect `plan.png` if it is attached. Do not give credit for S1 check 10 when the figure is generic, unreadable, or inconsistent with `plan.md`.
- Judge workflow quality, not whether the final reports are clinically correct.
- You may use deterministic evaluation metrics only as context about whether setup/validation likely happened.
- Do not score S4 or S5.

Assign E1-E5 failure codes where needed:
- E1 hallucination
- E2 resource error
- E3 logic error
- E4 code error
- E5 format/spec error

Respond with only JSON using this schema:
{
  "s1_plan_score": 0.0,
  "s1_rationale": "",
  "s1_failure": null,
  "s2_setup_score": 0.0,
  "s2_rationale": "",
  "s2_failure": null,
  "s3_validate_score": 0.0,
  "s3_rationale": "",
  "s3_failure": null,
  "detected_failure": null,
  "failure_explanation": "",
  "overall_rationale": ""
}
"""


MODEL_TOKENS = (
    "mlrg",
    "cxrmate",
    "cxrmate-ed",
    "hergen",
    "r2-llm",
    "qwen",
    "llava",
    "chexagent",
    "gpt",
    "claude",
    "gemini",
)


def _mean_binary_checks(values: list[bool | None]) -> float:
    applicable = [1.0 if value else 0.0 for value in values if value is not None]
    if not applicable:
        return 0.0
    return round(sum(applicable) / len(applicable), 4)


def _png_dimensions(path: str | None) -> tuple[int, int] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    data = file_path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def _encode_png_data_url(path: str | None) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    return "data:image/png;base64," + base64.b64encode(file_path.read_bytes()).decode("ascii")


def _artifact_output_dir(artifacts: dict) -> str:
    output_dir = artifacts.get("output_dir")
    if output_dir:
        return str(Path(output_dir).resolve())
    plan_md = artifacts.get("plan_md")
    if plan_md:
        return str(Path(plan_md).resolve().parent.parent)
    agent_outputs = artifacts.get("agent_outputs")
    if agent_outputs:
        return str(Path(agent_outputs).resolve().parent)
    return "<unknown>"


def _contains_any(text: str, tokens: tuple[str, ...] | list[str]) -> bool:
    return any(token in text for token in tokens)


def _missing_check_summary(step: str, checks: list[tuple[str, bool | None]]) -> str:
    applicable = [label for label, value in checks if value is not None]
    missing = [label for label, value in checks if value is False]
    passed = len(applicable) - len(missing)
    if not applicable:
        return f"{step}: no applicable checks."
    if not missing:
        return f"{step}: passed all {len(applicable)} applicable checks."
    return f"{step}: passed {passed}/{len(applicable)} applicable checks; missing {', '.join(missing[:4])}."


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _summarize_conversation(messages: list[dict], code_executions: list[dict]) -> str:
    snippets = []
    for msg in messages[-8:]:
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, list):
            content = json.dumps(content)
        content = (content or "").strip().replace("\n", " ")
        if content:
            snippets.append(f"{role}: {content[:220]}")
    if code_executions:
        snippets.append(f"code_executions={len(code_executions)}")
    return "\n".join(snippets)


def _safe_read_text(path: str | None, limit: int = 3000) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _summarize_code_executions(code_executions: list[dict], limit: int = 8) -> str:
    if not code_executions:
        return "No code executions recorded."
    snippets = []
    for entry in code_executions[:limit]:
        code_preview = " ".join((entry.get("code", "") or "").strip().split())[:180]
        stdout_preview = " ".join((entry.get("stdout_preview", "") or "").strip().split())[:120]
        snippets.append(
            f"turn={entry.get('turn')} exit={entry.get('exit_code')} lang={entry.get('language')} "
            f"code={code_preview!r} stdout={stdout_preview!r}"
        )
    if len(code_executions) > limit:
        snippets.append(f"... {len(code_executions) - limit} more executions omitted")
    return "\n".join(snippets)


def build_judge_user_prompt(conversation: dict, eval_report: dict, task: str) -> str:
    metrics = eval_report.get("metrics", {})
    fmt = eval_report.get("format", {})
    agg = eval_report.get("aggregate", {})
    code_executions = conversation.get("code_executions", [])
    conv_summary = _summarize_conversation(conversation.get("messages", []), code_executions)
    artifacts = conversation.get("artifact_paths", {})
    output_dir = _artifact_output_dir(artifacts)
    plan_excerpt = _safe_read_text(artifacts.get("plan_md"))
    plan_png_dims = _png_dimensions(artifacts.get("plan_png"))
    per_case = eval_report.get("_per_case", {})
    missing_cases = sorted([
        case_id for case_id, case_metrics in per_case.items()
        if case_metrics.get("missing_prediction")
    ])
    weak_cases = sorted(
        [
            (case_id, case_metrics.get("observation_f1", 0.0), case_metrics.get("report_similarity", 0.0))
            for case_id, case_metrics in per_case.items()
        ],
        key=lambda item: (item[1], item[2], item[0]),
    )[:4]

    return f"""Task: {task}
Tier: {conversation.get('tier', 'pro')}
Agent: {conversation.get('agent', 'unknown')}
Model: {conversation.get('model', 'unknown')}

Artifacts:
- output_dir: {output_dir}
- plan_md: {artifacts.get('plan_md')}
- plan_png: {artifacts.get('plan_png')}
- plan_png_dimensions: {plan_png_dims or "<missing>"}
- agent_outputs: {artifacts.get('agent_outputs')}

Path convention for this run:
- `plan/plan.md` == `{output_dir}/plan/plan.md`
- `plan/plan.png` == `{output_dir}/plan/plan.png`
- `agent_outputs/<case_id>/report.txt` == `{output_dir}/agent_outputs/<case_id>/report.txt`

Plan excerpt:
{plan_excerpt or "<missing>"}

Deterministic metrics:
- observation_f1: {metrics.get('observation_f1', 0.0)}
- report_similarity: {metrics.get('report_similarity', 0.0)}
- completion_rate: {fmt.get('completion_rate', 0.0)}
- output_valid: {fmt.get('output_format_valid', False)}
- current_rating: {agg.get('rating', 'F')}
- missing_prediction_cases: {missing_cases}
- weakest_cases: {weak_cases}

Code execution summary:
{_summarize_code_executions(code_executions)}

Conversation summary:
{conv_summary}
"""


def build_online_judge_user_content(conversation: dict, eval_report: dict, task: str) -> str | list[dict]:
    user_prompt = build_judge_user_prompt(conversation, eval_report, task)
    artifacts = conversation.get("artifact_paths", {})
    plan_png_url = _encode_png_data_url(artifacts.get("plan_png"))
    if not plan_png_url:
        return user_prompt
    return [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": plan_png_url}},
    ]


class BaseJudge(ABC):
    @abstractmethod
    def judge(self, conversation: dict, eval_report: dict, task: str) -> JudgeVerdict:
        raise NotImplementedError


class HeuristicJudge(BaseJudge):
    def __init__(self):
        self.model = "heuristic-report-judge"

    def judge(self, conversation: dict, eval_report: dict, task: str) -> JudgeVerdict:
        start = time.time()
        tier = conversation.get("tier", "pro")
        artifacts = conversation.get("artifact_paths", {})
        plan_md = Path(artifacts["plan_md"]) if artifacts.get("plan_md") else None
        plan_png = Path(artifacts["plan_png"]) if artifacts.get("plan_png") else None
        outputs_dir = Path(artifacts["agent_outputs"]) if artifacts.get("agent_outputs") else None

        plan_text = ""
        if plan_md and plan_md.is_file():
            plan_text = plan_md.read_text(encoding="utf-8", errors="replace").lower()
        plan_png_dims = _png_dimensions(artifacts.get("plan_png"))

        # S1
        has_plan_md = bool(plan_text)
        has_selected_model = _contains_any(plan_text, MODEL_TOKENS) or "decision:" in plan_text or "selected model" in plan_text
        has_selected_entrypoint = _contains_any(
            plan_text,
            (
                "official inference path",
                "inference path",
                "entrypoint",
                "from_pretrained",
                "main_v0926_ablation_study.py",
                "cxrmate.ipynb",
                "generate(",
            ),
        )
        states_case_unit = _contains_any(
            plan_text,
            (
                "one benchmark case is one study",
                "one case is one study",
                "one case is one full case input",
                "not one jpeg",
                "not one individual jpeg",
                "study-level",
                "each case is one study",
            ),
        )
        states_manifest_image_files = "manifest.json" in plan_text and _contains_any(
            plan_text,
            ("image_files", "all views", "all images"),
        )
        states_output_path = "agent_outputs" in plan_text and "report.txt" in plan_text
        requires_pilot_validation = _contains_any(
            plan_text,
            ("pilot", "single-case", "single case", "validation", "validate"),
        ) and _contains_any(plan_text, ("before full", "before batch", "s3"))
        defines_specific_pilot_check = (
            "report.txt" in plan_text
            or "image_files" in plan_text
            or "all views" in plan_text
            or "output path" in plan_text
        ) and _contains_any(plan_text, ("inspect", "check", "verify"))
        has_png = bool(plan_png and plan_png.is_file() and plan_png_dims)
        has_comparison = plan_text.count("|") >= 6 or _contains_any(
            plan_text,
            ("compare", "comparison", "candidate", "benchmark", "performance"),
        )
        has_justification = _contains_any(
            plan_text,
            ("justification", "reported", "performance", "blocked", "infeasible", "tradeoff", "chosen"),
        )
        s1_checks: list[tuple[str, bool | None]] = [
            ("plan.md", has_plan_md),
            ("selected model", has_selected_model),
            ("selected inference entrypoint", has_selected_entrypoint),
            ("case is one full case input, not one jpeg", states_case_unit),
            ("manifest image_files usage", states_manifest_image_files),
            ("exact report output path", states_output_path),
            ("pilot before batch", requires_pilot_validation),
            ("specific pilot check", defines_specific_pilot_check),
            ("plan.png present", has_png if tier in {"standard", "pro"} else None),
            ("plan.png workflow match", None if tier in {"standard", "pro"} else None),
            ("candidate comparison and justification", (has_comparison and has_justification) if tier in {"standard", "pro"} else None),
        ]
        s1 = _mean_binary_checks([value for _, value in s1_checks])
        if not has_plan_md:
            s1_failure = "E5"
        elif s1 < 0.5:
            s1_failure = "E3"
        else:
            s1_failure = None
        s1_rationale = _missing_check_summary("S1", s1_checks)
        if tier in {"standard", "pro"}:
            s1_rationale += " Heuristic mode checks `plan.png` presence but does not inspect its visual semantics."

        # S2
        code_executions = conversation.get("code_executions", [])
        successful = sum(1 for entry in code_executions if entry.get("exit_code") == 0)
        code_blob = "\n".join(entry.get("code", "").lower() for entry in code_executions)
        stdout_blob = "\n".join(entry.get("stdout_preview", "").lower() for entry in code_executions)
        trace_blob = code_blob + "\n" + stdout_blob
        repo_reference_signals = sum(
            token in code_blob
            for token in (
                "git clone https://github.com/",
                "cxrmate_repo/examples/cxrmate.ipynb",
                "examples/cxrmate.ipynb",
                "requirements.txt",
                "main_v0926_ablation_study.py",
                "run_cxr_",
            )
        )
        official_export_signals = sum(
            token in code_blob
            for token in (
                "split_and_decode_sections(",
                "decode_sections(",
                "section_parser",
                "postprocess",
                "export",
                "findings",
                "impression",
            )
        )
        official_entrypoint_signals = sum(
            token in code_blob
            for token in (
                "python3 main_v0926_ablation_study.py",
                "bash run_cxr_",
                "jupyter",
                "papermill",
                "python -m",
            )
        )
        custom_rewrite_signals = sum(
            token in code_blob
            for token in (
                "cat > \"$output_dir/inference.py\"",
                "cat > \"$output_dir/generate_reports.py\"",
                "inference_script = '''",
                "with open(script_path, \"w\")",
                "def generate_report(",
                "transforms.compose([",
                "model.generate(",
                "split_and_decode_sections(",
            )
        )
        custom_export_rewrite_signals = sum(
            token in code_blob
            for token in (
                "sep_token",
                "sep_token_id",
                "findings_ids = generated_ids[:sep_positions[0].item()]",
                "if len(findings) < 20",
                "return full_text",
                "generic split rule",
                "report.txt",
            )
        )
        output_cases = 0
        if outputs_dir and outputs_dir.is_dir():
            output_cases = sum(1 for child in outputs_dir.iterdir() if child.is_dir())
        faithful_pipeline = (
            repo_reference_signals >= 2
            and official_entrypoint_signals >= 1
            and custom_rewrite_signals == 0
            and custom_export_rewrite_signals == 0
        )
        wrapped_official_pipeline = repo_reference_signals >= 2 and custom_rewrite_signals >= 1
        custom_export_rewrite = custom_export_rewrite_signals >= 1 and official_export_signals == 0
        s2_checks: list[tuple[str, bool | None]] = [
            ("selected pipeline executed", successful >= 1 and _contains_any(code_blob, ("generate(", "from_pretrained", "python", "inference.py"))),
            ("manifest.json read", "manifest.json" in code_blob),
            ("case images loaded from image_files", "image_files" in code_blob and _contains_any(code_blob, ("image.open", "images", "pil"))),
            ("case kept as one input unit", _contains_any(code_blob, ("torch.stack", "pad_sequence", "study_images", "multi-view")) or "image_count" in trace_blob),
            ("report.txt written", "report.txt" in code_blob and output_cases >= 1),
            ("selected official inference entrypoint used", faithful_pipeline),
            ("selected official export logic used", official_export_signals >= 1 and not custom_export_rewrite),
        ]
        s2 = _mean_binary_checks([value for _, value in s2_checks])
        if successful == 0:
            s2_failure = "E4"
        elif s2 < 0.5:
            s2_failure = "E3"
        else:
            s2_failure = None
        s2_rationale = _missing_check_summary("S2", s2_checks)
        if custom_export_rewrite:
            s2_rationale += " The trace shows custom report-section handling instead of the method's export logic."
        elif wrapped_official_pipeline:
            s2_rationale += " The trace wraps the official method in custom benchmark code."
        elif custom_rewrite_signals > 0 and not faithful_pipeline:
            s2_rationale += " The trace includes custom inference code instead of the method's direct entrypoint."

        # S3
        messages_blob = " ".join(
            str(message.get("content", "")).lower()
            for message in conversation.get("messages", [])
        )
        ordered_trace = messages_blob + "\n" + trace_blob
        validate_signals = sum(
            token in ordered_trace
            for token in ("validate", "validation", "pilot", "first case", "single case", "check one case", "test one case")
        )
        pilot_run_detected = validate_signals >= 2
        pilot_output_written = pilot_run_detected and output_cases >= 1 and "report.txt" in code_blob
        pilot_report_opened = _contains_any(code_blob, ("report.txt", "cat ", "open(", "read_text(")) and _contains_any(messages_blob, ("inspect", "check", "verify", "pilot", "validate"))
        validation_checks_manifest = pilot_run_detected and "manifest.json" in code_blob and "image_files" in code_blob
        validation_checks_all_views = pilot_run_detected and _contains_any(ordered_trace, ("all views", "image_count", "image_files", "2 images", "3 images"))
        rerun_after_failure = False
        first_failure_turn = next((entry["turn"] for entry in code_executions if entry.get("exit_code") != 0), None)
        if first_failure_turn is not None:
            rerun_after_failure = any(
                entry.get("exit_code") == 0 and entry.get("turn", 0) > first_failure_turn
                for entry in code_executions
            )
        s3_checks: list[tuple[str, bool | None]] = [
            ("pilot run before batch", pilot_run_detected),
            ("pilot uses same pipeline as S2", pilot_run_detected and _contains_any(code_blob, ("inference.py", "generate(", "main_v0926_ablation_study.py"))),
            ("pilot writes report.txt", pilot_output_written),
            ("pilot report inspected", pilot_report_opened),
            ("validation checks image_files", validation_checks_manifest),
            ("validation confirms all case images", validation_checks_all_views),
            ("pilot rerun after failure", rerun_after_failure if first_failure_turn is not None else None),
        ]
        s3 = _mean_binary_checks([value for _, value in s3_checks])
        if pilot_run_detected:
            s3_failure = None if s3 >= 0.5 else "E3"
        else:
            s3_failure = "E3"
        s3_rationale = _missing_check_summary("S3", s3_checks)

        detected_failure = None
        failure_explanation = ""
        for step_name, score, failure in (
            ("S1", s1, s1_failure),
            ("S2", s2, s2_failure),
            ("S3", s3, s3_failure),
        ):
            if score < 0.5 and failure:
                detected_failure = f"{step_name}:{failure}"
                failure_explanation = f"{step_name} did not meet the rubric threshold."
                break

        overall = (
            "The agent's early workflow artifacts were "
            f"S1={s1:.1f}, S2={s2:.1f}, S3={s3:.1f} under the report-generation rubric."
        )
        return JudgeVerdict(
            s1_plan_score=s1,
            s1_rationale=s1_rationale,
            s1_failure=s1_failure,
            s2_setup_score=s2,
            s2_rationale=s2_rationale,
            s2_failure=s2_failure,
            s3_validate_score=s3,
            s3_rationale=s3_rationale,
            s3_failure=s3_failure,
            detected_failure=detected_failure,
            failure_explanation=failure_explanation,
            overall_rationale=overall,
            judge_model=self.model,
            judge_backend="heuristic",
            judge_latency_s=round(time.time() - start, 3),
        )


class OnlineJudge(BaseJudge):
    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _chat_completion(self, system: str, user: str | list[dict]) -> tuple[dict, int, int]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 1200,
        }
        request = urllib.request.Request(
            url=self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
        usage = data.get("usage", {})
        return data, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    def judge(self, conversation: dict, eval_report: dict, task: str) -> JudgeVerdict:
        start = time.time()
        user_content = build_online_judge_user_content(conversation, eval_report, task)
        data, input_tokens, output_tokens = self._chat_completion(JUDGE_SYSTEM_PROMPT, user_content)
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        payload = _extract_json(content)
        verdict = JudgeVerdict(
            s1_plan_score=float(payload.get("s1_plan_score", 0.0)),
            s1_rationale=payload.get("s1_rationale", ""),
            s1_failure=payload.get("s1_failure"),
            s2_setup_score=float(payload.get("s2_setup_score", 0.0)),
            s2_rationale=payload.get("s2_rationale", ""),
            s2_failure=payload.get("s2_failure"),
            s3_validate_score=float(payload.get("s3_validate_score", 0.0)),
            s3_rationale=payload.get("s3_rationale", ""),
            s3_failure=payload.get("s3_failure"),
            detected_failure=payload.get("detected_failure"),
            failure_explanation=payload.get("failure_explanation", ""),
            overall_rationale=payload.get("overall_rationale", ""),
            judge_model=self.model,
            judge_backend="online",
            judge_latency_s=round(time.time() - start, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return verdict


def _extract_keys(path: str | Path) -> list[str]:
    return load_api_keys(path)


def create_judge(online: bool = False, config_path: str | None = None, **kwargs) -> BaseJudge:
    if not online:
        return HeuristicJudge()

    config_path = config_path or "eval_report_gen/agent_config.yaml"
    config = load_config(config_path)
    judge_cfg = config.get("judge", {})
    api_key_file = kwargs.get("api_key_file") or judge_cfg.get("api_key_file") or config.get("api_key_file")
    keys = _extract_keys(api_key_file)
    if not keys:
        raise RuntimeError(f"No API keys found in {api_key_file}")
    return OnlineJudge(
        model=kwargs.get("model") or judge_cfg["online_model"],
        base_url=kwargs.get("base_url") or judge_cfg["base_url"],
        api_key=kwargs.get("api_key") or keys[0],
    )
