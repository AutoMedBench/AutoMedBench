#!/usr/bin/env python3
"""Real benchmark runner for eval_report_gen.

The agent gets one tool, `execute_code`, plus `submit_results`.
Baseline agents are also supported for local validation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_io import load_config
from detail_report import generate_detail_report, print_detail_report
from format_checker import check_submission
from llm_judge import create_judge
from make_dummy_agents import build_baseline
from run_eval import run_eval
from summary_plots import generate_summary_plots
from task_loader import (
    discover_cases,
    get_task_data_root,
    load_model_info,
    load_requirements_path,
    load_skill,
    load_task_config,
)
from tier_config import get_tier_config

try:
    from api_key_loader import load_api_keys
except ModuleNotFoundError:  # pragma: no cover - import path depends on invocation style
    from eval_report_gen.api_key_loader import load_api_keys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "agent_config.yaml"


def _extract_keys(path: str | Path) -> list[str]:
    return load_api_keys(path)


def _chat_completion_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return base_url + "/chat/completions"


def call_api(
    api_key: str,
    model: str,
    base_url: str,
    system: str,
    messages: list[dict],
    tools: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 4000,
    extra_api_keys: list[str] | None = None,
    max_retries: int = 6,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    candidate_keys = [api_key] + [key for key in (extra_api_keys or []) if key and key != api_key]
    last_error: Exception | None = None
    for attempt in range(max_retries):
        current_key = candidate_keys[attempt % len(candidate_keys)] if candidate_keys else api_key
        request = urllib.request.Request(
            url=_chat_completion_url(base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {current_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt))
                continue
            raise
        except Exception as error:
            last_error = error
            raise
    else:
        raise last_error or RuntimeError("API call failed without an explicit error")

    choice = data["choices"][0]
    message = choice["message"]
    tool_calls = []
    for tool_call in message.get("tool_calls", []) or []:
        arguments = tool_call["function"]["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        tool_calls.append(
            {
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "arguments": arguments,
            }
        )
    usage = data.get("usage", {})
    return {
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
        "finish_reason": choice.get("finish_reason", "stop"),
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }


def execute_code(language: str, code: str, cwd: str | Path, timeout: int, env: dict[str, str]) -> dict:
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    suffix = ".py" if language == "python" else ".sh"

    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, dir=cwd) as handle:
        script_path = Path(handle.name)
        handle.write(code)

    if language == "python":
        cmd = [sys.executable, str(script_path)]
    elif language == "bash":
        cmd = ["bash", str(script_path)]
    else:
        return {"exit_code": -2, "stdout": "", "stderr": f"Unsupported language: {language}"}

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env={**os.environ, **env},
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _artifact_paths(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    return {
        "output_dir": str(output_dir),
        "plan_md": str((output_dir / "plan" / "plan.md").resolve()),
        "plan_png": str((output_dir / "plan" / "plan.png").resolve()),
        "agent_outputs": str((output_dir / "agent_outputs").resolve()),
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run Python or bash code in the benchmark workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "bash"]},
                    "code": {"type": "string"},
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_results",
            "description": "Signal that all report.txt outputs are ready for validation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def _tier_skill_bundle(task_id: str, tier_name: str) -> str:
    skill_map = {
        "lite": ["lite_s1.md", "lite_s2.md", "lite_s3.md"],
        "standard": ["standard_s1.md", "lite_s2.md", "standard_s3.md"],
        "pro": [],
    }
    sections = []
    for filename in skill_map.get(tier_name, []):
        content = load_skill(task_id, filename).strip()
        if content:
            sections.append(content)
    return "\n\n".join(sections)


def build_tier_system_prompt(
    tier_config,
    task_config: dict,
    model_info: dict,
    data_dir: str,
    output_dir: str,
) -> str:
    lite = model_info.get("lite", {})
    standard = model_info.get("standard", {})
    pro = model_info.get("pro", {})
    task_id = task_config["_task_id"]
    requirements_path = load_requirements_path(task_id)
    tier_skills = _tier_skill_bundle(task_id, tier_config.name)
    prompt = f"""You are a medical AI coding agent for a chest X-ray report-generation benchmark.

Task: {task_config['task_description']}

Public data directory:
{Path(data_dir).resolve()}

Each case lives under:
  <case_id>/images/*.jpg
  <case_id>/manifest.json

Required outputs under:
{Path(output_dir).resolve()}

All relative output paths below are relative to `OUTPUT_DIR`.

Output contract:
  agent_outputs/<case_id>/report.txt
  plan/plan.md

Workflow:
  S1 PLAN      research and record the reporting pipeline
  S2 SETUP     establish the generation pipeline and dependencies
  S3 VALIDATE  test one or more pilot studies before batch generation
  S4 INFERENCE generate reports for all cases
  S5 SUBMIT    verify outputs and call submit_results

Helpful environment variables inside execute_code:
  REPORT_GEN_API_KEY
  REPORT_GEN_BASE_URL
  REPORT_GEN_MODEL_HINT
  PUBLIC_DATA_DIR
  OUTPUT_DIR

Execution rules:
- `PUBLIC_DATA_DIR` and `OUTPUT_DIR` are absolute filesystem paths. Use them directly.
- Do not prepend repo-relative prefixes such as `eval_report_gen/...` when writing outputs.
- Each `execute_code` call runs in a fresh subprocess. Python variables and functions do not persist across tool calls.
- If you need reusable logic, write a script under `OUTPUT_DIR` and invoke it later.
- Spend at most 2 `execute_code` calls on environment inspection before implementing the pipeline.
- Use every image listed in `manifest.json` for a study. Do not silently drop extra views.
- Prefer an open-source report-generation model or pipeline when the tier guidance says so.
- In S1, prioritize the best-supported model choice by reported benchmark performance and task fit, not by convenience.
- Do not justify choosing a weaker model because it is easier to access via HuggingFace, simpler to wire, or faster to prototype.
- If you do not choose the strongest candidate in the model range, explicitly justify the tradeoff with evidence beyond implementation convenience.
- Once you choose a method, follow this checklist in order:
  1. find the official inference entrypoint;
  2. recover the official environment/dependency versions for that method;
  3. find the official output/export contract for that method;
  4. reuse those official semantics when mapping model output to benchmark output;
  5. only if no official export path exists, write the smallest necessary benchmark adapter.
- Before attempting inference, inspect the selected method's README, notebook, generation_config.json,
  config.json, requirements.txt, or lockfile to recover intended versions for transformers, torch,
  torchvision, tokenizers, and remote-code dependencies.
- Before creating scripts or environments, explicitly probe which interpreters and installers exist
  on this machine (for example: `python`, `python3`, `python3.9`, `python3.10`, `python3.11`,
  `python3.12`, `pip`, `pip3`). Do not assume `python` exists.
- If the selected method needs a versioned environment that is not already available, first try to
  recover or create that environment yourself using the interpreters that actually exist on the
  machine; only after that fails should you treat the environment as a blocker.
- Any shell or Python launcher you write should use a verified interpreter path discovered during
  setup, not an assumed executable name.
- If the model config declares a transformers_version, prefer matching that version in a clean env
  before attempting ad-hoc code patches.
- Do not assume your own generic findings/impression split rule if the selected method defines a different output structure, section helper, or post-processing recipe.
- Only adapt dataset I/O and output formatting to match this benchmark's contract.
- Do not replace the selected method with a custom approximate implementation, custom decoding recipe, or rewritten preprocessing pipeline unless the official inference path is unavailable.
- Do not replace the selected method's official export logic with a custom section-extraction, truncation, or cleanup rule unless the official path does not provide one.
- If you must deviate from the official pipeline, document exactly why and treat that path as a fallback rather than as a faithful reproduction of the method.
- If you must deviate from the official export logic, document exactly why, what was changed, and how the benchmark-facing report text is derived from the model's raw output.
- If the chosen open-source model does not match this benchmark's input/output contract exactly, document the adaptation in `OUTPUT_DIR/plan/plan.md`.
- For this benchmark, the clinical evaluation reference is findings-only text.

Tier: {tier_config.name}
"""
    if tier_config.name == "lite":
        prompt += f"\nRecommended model: {lite.get('model_architecture', '')}\n{lite.get('model_description', '')}\n"
    elif tier_config.name == "standard":
        prompt += "\nCandidate model families:\n"
        for item in standard.get("model_range", []):
            prompt += f"- {item}\n"
        prompt += f"\n{standard.get('model_description', '')}\n"
    else:
        prompt += f"\nClinical background:\n{pro.get('clinical_background', '')}\n"
    if tier_config.provide_requirements_txt and requirements_path:
        prompt += f"\nRequirements file:\n- {Path(requirements_path).resolve()}\n"
    if tier_skills:
        prompt += f"\nTier-specific workflow guidance:\n{tier_skills}\n"
    return prompt


KICKOFF = {
    "lite": "Start at S1. Inspect the available cases, compare the open-source model candidates, choose the best-supported report-generation model and checkpoint based on reported performance and task fit, write OUTPUT_DIR/plan/plan.md, then set up the environment and validate on a pilot study before batch generation. Do not choose a weaker model just because it is easier to use. After selecting a method, first find its official inference path, then recover its intended dependency versions, then find its official output/export contract, and only if no official export path exists should you write a minimal benchmark adapter. Before setup scripts, explicitly detect which Python and pip executables exist on this machine, use a verified interpreter path rather than assuming `python`, and try to recover the required environment yourself before declaring an environment blocker. If the model declares a transformers_version, match it before patching remote code.",
    "standard": "Start at S1. Compare at least three open-source report-generation candidates from the model range, record a comparison table in OUTPUT_DIR/plan/plan.md, create OUTPUT_DIR/plan/plan.png, and choose the strongest method by public evidence and task fit. Do not choose a weaker method because it is easier to set up. If the strongest method is blocked by gated access or missing official assets, document that blocker and select the next-best runnable option. Then follow this order: official inference path, official output/export contract, minimal benchmark adapter only if needed; validate on a pilot study, and only then run the full batch. Before any setup script, explicitly probe available Python and pip executables, use a verified interpreter path instead of assuming `python`, and attempt to recover the required runtime yourself before concluding the environment is a blocker.",
    "pro": "Start at S1. Do broad research, justify the best reporting pipeline, create plan artifacts under OUTPUT_DIR/plan/, validate on pilot studies, then finish the batch.",
}


class BenchmarkRunner:
    def __init__(
        self,
        agent_name: str,
        task: str,
        tier: str = "pro",
        llm_judge: bool = True,
        online_judge: bool = False,
        output_dir: str | None = None,
    ):
        self.tier = get_tier_config(tier)
        self.task_config = load_task_config(task)
        self.task_id = self.task_config["_task_id"]
        self.model_info = load_model_info(self.task_id)
        self.cases = discover_cases(self.task_id)
        self.data_root = Path(get_task_data_root(self.task_id)).resolve()
        self.public_dir = (self.data_root / "public").resolve()
        self.private_dir = (self.data_root / "private").resolve()
        self.llm_judge = llm_judge
        self.online_judge = online_judge

        self.config = load_config(CONFIG_PATH)
        self.agent_cfg = self.config["agents"][agent_name]
        self.agent_name = agent_name
        self.model = self.agent_cfg.get("model", agent_name)
        self.max_turns = int(self.config["settings"]["max_turns"])
        self.max_time_s = int(self.task_config.get("time_limit_s", self.config["settings"]["max_time_s"]))
        self.temperature = float(self.agent_cfg.get("temperature", self.config["settings"]["temperature"]))
        self.agent_max_tokens = int(self.agent_cfg.get("max_tokens", self.config["settings"].get("agent_max_tokens", 4000)))

        run_tag = time.strftime("%y%m%d") + "-" + "".join(random.choices("0123456789abcdef", k=6))
        if output_dir:
            self.run_dir = (Path(output_dir) / run_tag).resolve()
        else:
            self.run_dir = (SCRIPT_DIR / "runs" / self.tier.name / agent_name / self.task_id / run_tag).resolve()
        self.process_dir = (self.run_dir / "process").resolve()
        self.output_dir = (self.run_dir / "outputs").resolve()
        self.agent_outputs_dir = (self.output_dir / self.task_config["agent_output_subdir"]).resolve()
        self.plan_dir = (self.output_dir / self.task_config["plan_subdir"]).resolve()
        self.process_dir.mkdir(parents=True, exist_ok=True)
        self.agent_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.plan_dir.mkdir(parents=True, exist_ok=True)

        self.system_prompt = build_tier_system_prompt(
            self.tier,
            self.task_config,
            self.model_info,
            data_dir=str(self.public_dir),
            output_dir=str(self.output_dir),
        )
        (self.output_dir / "tier_prompt.txt").write_text(self.system_prompt, encoding="utf-8")

        api_key_file = self.agent_cfg.get("api_key_file") or self.config.get("api_key_file")
        self.api_keys = _extract_keys(api_key_file) if api_key_file else []
        inline_api_key = self.agent_cfg.get("api_key") or self.config.get("api_key") or ""
        if not self.api_keys and inline_api_key:
            self.api_keys = [str(inline_api_key)]
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.base_url = self.agent_cfg.get("base_url") or self.config.get("default_base_url", "")

    def _run_baseline(self) -> tuple[list[dict], list[dict], int, int, int]:
        build_baseline(self.task_id, self.agent_cfg["baseline"], self.output_dir)
        conversation = [
            {"role": "user", "content": f"Built baseline '{self.agent_cfg['baseline']}' for {len(self.cases)} cases."}
        ]
        code_executions = []
        return conversation, code_executions, 0, 0, 0

    def _run_agent(self) -> tuple[list[dict], list[dict], int, int, int]:
        messages = [{"role": "user", "content": KICKOFF[self.tier.name]}]
        code_executions: list[dict] = []
        total_in = 0
        total_out = 0
        submitted = False
        submit_calls = 0
        start = time.time()

        trace_path = self.process_dir / "trace.jsonl"
        tool_log_path = self.process_dir / "tool_calls.jsonl"
        api_log_path = self.process_dir / "api_responses.jsonl"
        with (
            trace_path.open("w", encoding="utf-8") as trace_handle,
            tool_log_path.open("w", encoding="utf-8") as tool_log,
            api_log_path.open("w", encoding="utf-8") as api_log,
        ):
            for turn in range(self.max_turns):
                if time.time() - start > self.max_time_s:
                    break

                response = call_api(
                    api_key=self.api_key,
                    extra_api_keys=self.api_keys[1:],
                    model=self.model,
                    base_url=self.base_url,
                    system=self.system_prompt,
                    messages=messages,
                    tools=TOOLS,
                    temperature=self.temperature,
                    max_tokens=self.agent_max_tokens,
                )
                total_in += response["input_tokens"]
                total_out += response["output_tokens"]
                api_log.write(json.dumps({
                    "turn": turn + 1,
                    "request": {
                        "model": self.model,
                        "base_url": self.base_url,
                        "temperature": self.temperature,
                        "max_tokens": self.agent_max_tokens,
                        "message_count": len(messages),
                        "messages": messages,
                        "tools": TOOLS,
                    },
                    "response": response,
                }) + "\n")
                api_log.flush()
                trace_handle.write(json.dumps({
                    "turn": turn + 1,
                    "type": "api_call",
                    "tool_calls": [call["name"] for call in response["tool_calls"]],
                    "finish_reason": response["finish_reason"],
                    "content": response["content"],
                    "content_preview": response["content"][:200],
                }) + "\n")
                trace_handle.flush()

                assistant_message = {"role": "assistant", "content": response["content"] or ""}
                if response["tool_calls"]:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["arguments"]),
                            },
                        }
                        for tool_call in response["tool_calls"]
                    ]
                messages.append(assistant_message)

                if not response["tool_calls"]:
                    break

                for tool_call in response["tool_calls"]:
                    if tool_call["name"] == "execute_code":
                        arguments = tool_call.get("arguments", {}) or {}
                        language = arguments.get("language")
                        code = arguments.get("code")
                        if not language or not code:
                            result = {
                                "error": (
                                    "Invalid or truncated execute_code call. "
                                    "Provide JSON arguments with both 'language' and 'code'."
                                )
                            }
                        else:
                            env = {
                                "REPORT_GEN_API_KEY": self.api_key,
                                "REPORT_GEN_BASE_URL": self.base_url,
                                "REPORT_GEN_MODEL_HINT": self.model,
                                "PUBLIC_DATA_DIR": str(self.public_dir),
                                "OUTPUT_DIR": str(self.output_dir),
                            }
                            result = execute_code(
                                language=language,
                                code=code,
                                cwd=self.output_dir,
                                timeout=max(60, int(self.max_time_s - (time.time() - start))),
                                env=env,
                            )
                            code_executions.append({
                                "turn": turn + 1,
                                "language": language,
                                "code": code,
                                "exit_code": result["exit_code"],
                                "stdout_preview": result["stdout"][:200],
                            })
                    elif tool_call["name"] == "submit_results":
                        result = check_submission(
                            agent_dir=self.agent_outputs_dir,
                            case_ids=self.cases,
                            task_config=self.task_config,
                        )
                        submitted = True
                        submit_calls += 1
                    else:
                        result = {"error": f"Unknown tool {tool_call['name']}"}

                    tool_log.write(json.dumps({
                        "turn": turn + 1,
                        "tool": tool_call["name"],
                        "arguments": tool_call["arguments"],
                        "result": result,
                    }) + "\n")
                    tool_log.flush()
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result),
                    })

                if submitted:
                    break

        return messages, code_executions, total_in, total_out, submit_calls

    def run(self) -> dict:
        run_log = self.run_dir / "run.log"
        original_stdout = sys.stdout
        run_log_handle = run_log.open("w", encoding="utf-8")

        class Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for stream in self.streams:
                    stream.write(data)
                    stream.flush()

            def flush(self):
                for stream in self.streams:
                    stream.flush()

        sys.stdout = Tee(original_stdout, run_log_handle)
        start = time.time()
        print(f"\n==============================================================")
        print(f"  eval_report_gen -- {self.tier.name.upper()} tier")
        print(f"  Agent: {self.agent_name} | Task: {self.task_id} | Cases: {len(self.cases)}")
        print(f"  Output: {self.run_dir}")
        print(f"==============================================================\n")

        if self.agent_cfg.get("kind") == "baseline":
            messages, code_executions, total_in, total_out, submit_calls = self._run_baseline()
        else:
            messages, code_executions, total_in, total_out, submit_calls = self._run_agent()

        wall_time = round(time.time() - start, 3)

        conversation = {
            "agent": self.agent_name,
            "model": self.model,
            "task": self.task_id,
            "tier": self.tier.name,
            "messages": messages,
            "code_executions": code_executions,
            "artifact_paths": _artifact_paths(self.output_dir),
        }
        conv_path = self.process_dir / "conversation.json"
        conv_path.write_text(json.dumps(conversation, indent=2), encoding="utf-8")

        eval_report = run_eval(
            gt_dir=str(self.private_dir),
            agent_dir=str(self.agent_outputs_dir),
            case_ids=self.cases,
            task_config=self.task_config,
            llm_judge=self.llm_judge,
            online_judge=self.online_judge,
            conversation=conversation,
        )

        runtime = {
            "wall_time_s": wall_time,
            "api_calls": 0 if self.agent_cfg.get("kind") == "baseline" else len([m for m in messages if m.get("role") == "assistant"]),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "estimated_cost_usd": 0.0,
            "code_executions": len(code_executions),
        }
        if self.agent_name in self.config.get("pricing", {}) and total_in + total_out:
            pricing = self.config["pricing"][self.agent_name]
            runtime["estimated_cost_usd"] = round(
                (total_in * pricing["input_per_1m"] + total_out * pricing["output_per_1m"]) / 1_000_000,
                4,
            )

        tool_summary = {
            "total": len(code_executions) + submit_calls,
            "by_tool": {
                "execute_code": len(code_executions),
                "submit_results": submit_calls,
            },
            "errors": sum(1 for entry in code_executions if entry["exit_code"] != 0),
        }

        judge_verdict = eval_report.get("llm_judge")
        if isinstance(judge_verdict, dict) and "error" in judge_verdict:
            judge_verdict = None

        detail = generate_detail_report(
            eval_report=eval_report,
            runtime=runtime,
            agent_name=self.agent_name,
            model=self.model,
            task=self.task_id,
            tool_summary=tool_summary,
            judge_verdict=judge_verdict,
            tier=self.tier.name,
        )

        report_path = self.run_dir / "detail_report.json"
        report_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
        if self.tier.generate_summary_plots:
            plot_dir = self.process_dir / "plots"
            generate_summary_plots(detail, plot_dir)

        print_detail_report(detail)
        print(f"[Runner] Report -> {report_path}")
        print(f"[Runner] Conversation -> {conv_path}")

        sys.stdout = original_stdout
        run_log_handle.close()
        return detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval_report_gen benchmark")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--task", default="mimic-cxr-report-task")
    parser.add_argument("--tier", default="pro", choices=["lite", "standard", "pro"])
    parser.add_argument("--offline-judge", action="store_true", help="Use heuristic judge instead of online judge")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    runner = BenchmarkRunner(
        agent_name=args.agent,
        task=args.task,
        tier=args.tier,
        llm_judge=True,
        online_judge=not args.offline_judge,
        output_dir=args.output_dir,
    )
    runner.run()


if __name__ == "__main__":
    main()
