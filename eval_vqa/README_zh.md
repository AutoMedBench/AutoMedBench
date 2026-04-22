# MedAgentsBench VQA Benchmark (V2)

医学视觉问答（VQA）agent 评测套件。与 `eval_seg/` 同构，采用 **single-LLM /
execute_code-only** 的 coding-agent 范式：agent 在一个长对话里自己写 Python
完成 S1–S5 全流程，框架不提供额外工具。

覆盖数据集：**PathVQA**、**VQA-RAD**、**SLAKE-EN**、**MedFrameQA**、**MedXpertQA-MM**。

---

## 1. 快速开始

### 1.1 环境

```bash
conda activate base
cd eval_vqa
pip install -r requirements.txt
```

必要环境变量：`NVDA_API_KEY` 或 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`（按 agent
而定），`HF_TOKEN`（下载受门控模型），`OPENROUTER_API_KEY`（LLM-as-judge，默认
Claude Haiku 4.5）。

### 1.2 单次基准

```bash
python -u eval_vqa/benchmark_runner.py \
  --agent nvda-claude-opus-4-6 \
  --task pathvqa-task \
  --tier lite \
  --subset all \
  --output-dir ./runs/<experimenter>/vqa-<short>-<task>-<tier>
```

- `--task` ∈ `pathvqa-task` / `vqa-rad-task` / `slake-task` / `medframeqa-task` /
  `medxpertqa-mm-task`
- `--tier` ∈ `lite`（固定模型 `microsoft/llava-med-v1.5-mistral-7b`）/ `standard`
  （五候选开放 VLM）
- `--subset` ∈ `all` / `smoke`（10 条）/ `calibration`（15 条）
- `--output-dir` 会自动追加 `<YYMMDD>-<6hex>`，避免并行重名

### 1.3 纯评分（已有 predictions）

```bash
python eval_vqa/run_eval.py \
  --gt-dir    <staged_private>/<qid>/ \
  --agent-dir <run_root>/predictions/ \
  --public-dir <staged_public>/ \
  --task pathvqa-task --tier lite \
  --enable-answer-judge \
  --answer-judge-model anthropic/claude-haiku-4.5
```

### 1.4 Sweep（多 agent × 多 repeat × 多 GPU）

```bash
python eval_vqa/sweep.py \
  --agents nvda-claude-opus-4-6,gpt-5.4,gemini-3.1-pro \
  --repeats 3 --parallel-workers 4 \
  --gpu-devices 0,1,2,3 \
  --task pathvqa-task --tier lite \
  --sweep-root /data2/<experimenter>/vqa-sweep-<name>
```

**`/dev/shm` 预检**（BUG-044 派生）：并行 workers > 1 时会按
`max(4 GiB, 4 × workers)` 核对 `/dev/shm` 剩余空间。不足时直接退出并给出缓解
建议：降 workers、docker `--shm-size=8g`、或 `--allow-low-shm` 绕过。

---

## 2. 目录结构

```
eval_vqa/
├── benchmark_runner.py      # agent loop 主入口（单 LLM + execute_code）
├── run_eval.py              # 评分入口（可独立运行）
├── sweep.py                 # 并行 sweep 驱动（含 /dev/shm 预检）
├── aggregate.py             # workflow_score / step_scores / 评级聚合
├── vqa_scorer.py            # 样本级 EM/F1/yes-no/judge 打分
├── answer_judge.py          # LLM-as-judge（Claude Haiku via OpenRouter）
├── llm_judge.py             # workflow-level S1-S3 启发式 judge
├── failure_classifier.py    # E1-E8 错误码分类
├── inference_verifier.py    # smoke_forward / model_call / postprocess 验证
├── tool_call_recovery.py    # minimax/qwen XML tool_call 降级恢复（BUG-043/040）
├── format_checker.py        # submission schema 校验
├── tier_config.py           # lite / standard 差分 + step_weights
├── task_loader.py           # tasks/ 下的 config.yaml / model_info.yaml
├── stage_hf_datasets.py     # 公私 manifest 分离（隐私合约）
├── medal_tier.py            # 奖牌分级
├── detail_report.py         # 人类可读报告
├── prompts/                 # 五阶段基础 prompt（见 §3）
│   ├── s1_plan/{lite,standard}.md
│   ├── s2_setup/lite.md
│   ├── s3_validate/lite_standard.md
│   ├── s4_inference/all.md
│   └── s5_submit/all.md
├── tasks/<task-id>/         # 任务配置 + 每阶段 skill 片段
│   ├── config.yaml          # answer_mode / valid_labels / subsets …
│   ├── model_info.yaml      # lite_model / standard_candidates
│   ├── lite_s1.md / lite_s2.md / lite_s3.md
│   └── standard_s1.md / standard_s3.md
└── tests/                   # pytest 单测（100+ 用例）
```

---

## 3. Workflow 五阶段

Agent 在一个长对话里依次推进 S1 → S5，全程只用 `execute_code`。
阶段权重 **S1 0.25 / S2 0.15 / S3 0.35 / S4 0.15 / S5 0.10**。

以下每个阶段给出：**定义 → artefact 契约 → prompt 全文 → 打分规则**。

### 3.1 S1 — PLAN（权重 0.25）

**定义**：产出 `plan/plan.md`，说明选型、数据契约、label/answer 提取方式、
smoke 计划。

**Prompt（lite）**：

```markdown
Use `UCSC-VLAA/MedVLThinker-3B-RL_m23k`. Write `plan.md` covering:

- local model loading path
- staged `question.json` / `answer.json` contract
- label extraction into `predicted_label`
- smoke validation on `1-10` samples before any larger run
```

**Prompt（standard）**：agent 需在五候选（MedVLThinker / MedVLSynther /
Qwen2.5-VL-3B / gemma-4-E2B / gemma-4-E4B）中选型，记录拒选原因，并标注
gated 风险。

**打分**：默认 `None`；启用 `--llm-judge` 时由 `llm_judge.py` 给 0–1 分。
未启用 judge 时，S1 不进入 workflow 分母（renormalization，BUG-review #2）。

### 3.2 S2 — SETUP（权重 0.15）

**定义**：装依赖、下载模型、跑一次 GPU forward pass，落盘
`smoke_forward.json`。

**Prompt**：

```markdown
Install dependencies from the task `requirements.txt`. For PyTorch packages,
use the official wheel index and choose a CUDA build that matches the host
driver instead of assuming the newest wheel will run. Then download model and
processor assets locally, and confirm at least one successful forward pass on
GPU before continuing.

**Required artefact — `${WORKSPACE_DIR}/smoke_forward.json`.** After your
smoke forward pass, write this file with the following schema:

{
  "model_name": "<HF repo id you actually loaded>",
  "device": "cuda" | "cuda:0" | "...",
  "wall_s": <float, end-to-end seconds, must be >= 0.3>,
  "raw_output_sample": "<first ~200 chars of the raw decoded model text>",
  "success": true
}

The scorer reads this artefact directly. If `smoke_forward.json` is missing,
`success` is not `true`, `wall_s < 0.3`, or `raw_output_sample` is empty /
starts with `heuristic:` / `fallback:` / `placeholder:`, **S2 will be forced
to 0.0 regardless of any other evidence**.
```

**打分（三元 binary，P3）**：

```
S2 = mean(env_ready, model_loaded, smoke_forward_passed)
```

- `env_ready`：agent 至少写出一个 valid output 或 `completion_rate > 0`
- `model_loaded`：trace 里有 `model.generate()` / `from_pretrained()` 证据
- `smoke_forward_passed`：`inference_verifier.check_smoke_forward()` 通过

### 3.3 S3 — VALIDATE（权重 0.35，最高权重）

**定义**：在 15 条样本上 calibrate；产出 `answer_postprocess.py`（短答案
normalization）+ `s3_calibration.json`（含 gold 对照）。

**Prompt 全文**（`prompts/s3_validate/lite_standard.md`）：

```markdown
Run inference on **exactly 15 samples** from the public split as a calibration
pass. Design a deterministic post-processor before S4 starts at scale.

**Turn budget — commit, don't loop.** S3 calibration should take no more than
10 `execute_code` turns. Partial calibration is better than a stalled run with
zero `answer.json` files.

**Required artefact 1 — `${WORKSPACE_DIR}/answer_postprocess.py`**:

def postprocess(raw: str) -> str:
    """Return the short normalized answer string from raw VLM text."""

Must be importable without side effects, never raise, return str.

**Required artefact 2 — `${WORKSPACE_DIR}/s3_calibration.json`** (≥15 records):

{"question_id": "<id>",
 "raw_model_output": "<full decoded text>",
 "predicted_answer": "<postprocess(raw_model_output)>",
 "gold_answer": "<reference answer>",
 "hit": true|false}

**Anti-cheat rules (enforced):**
- `gold_answer` 必须来自真实 reference；写 "unknown" / "n/a" / "" 伪造 hit
  会触发 `invalid_gold_rate > 0.2 → S3 cap 0.5`
- `raw_model_output` 长度 ≥ 5，不得以标点开头；`malformed_raw_rate > 0.2 →
  S3 cap 0.5`（BUG-042，LLaVA-Med 解码截断典型症状）

**Scorer 强制：**
- `answer_postprocess.py` 缺失 / 不可 import / 无 callable → `postprocess_valid=False`
  → S3 cap 0.5，`failure=E3`
- `s3_calibration.json` 缺失 / <15 records / 畸形 → 同上
- 15 turn 预警（BUG-044）：任一 artefact 未落盘会收到 `[S3 BUDGET WARNING]`
  提示立即产出并进入 S4
```

**打分**：
- 默认 `None`（需 llm_judge 填充）
- `postprocess_valid=False` → hard cap 0.5
- `inference_mode ∈ {heuristic, placeholder, empty}` → cap 0.2（防止 judge
  对 fake outputs 打高分）

### 3.4 S4 — INFERENCE（权重 0.15）

**定义**：在全部 question_ids 上跑真实 VLM inference；每条写
`<qid>/answer.json`；模型只加载一次。

**Prompt 要点**（全文见 `prompts/s4_inference/all.md`）：

- **Load the model once**：禁止 per-sample `from_pretrained`，禁止
  subprocess 循环重载
- **Real inference required**：`raw_model_output` 必须来自 `.generate()`；
  写 `heuristic:` / `fallback:` / `placeholder:` / `unknown` / 空串会被视为
  placeholder
- **Use the S3 post-processor**：必须
  `from answer_postprocess import postprocess`，不得在 S4 重造规则
- **Open-ended 短答案契约**（BUG-047 + review #7 新增）：对
  `answer_mode=open_ended` 的任务（PathVQA / VQA-RAD / SLAKE），system
  prompt 顶层注入 "Open-ended answer contract" 区块：
  - `predicted_answer` ≤ 5 词，VLM prose 必须被 postprocess 收敛
  - Yes/no 仅输出 `yes` / `no`
  - 无前导冠词、无句末标点、无解释

**打分（P1-B real-but-broken 敏感）**：

```
base = 0.5 · completion_rate + 0.5 · parse_rate

caps（依次应用）：
- placeholder_rate > 0.05    → cap 0.2
- model_call_detected = False → cap 0.3
- completion ≥ 0.99 ∧ placeholder ≤ 0.05 ∧ model_call
  ∧ accuracy < 0.05          → cap 0.5  (real_but_broken)
```

### 3.5 S5 — SUBMIT（权重 0.10）

**定义**：提交前核对完整性、schema 合法。

**Prompt**：

```markdown
Before submission, verify completeness, parseability, and schema validity for
all expected prediction records.
```

**打分**：

```
S5 = 0.5 · has_valid_results + 0.5 · submission_format_valid
```

- `has_valid_results`：至少一条 valid output
- `submission_format_valid`：`format_checker.check_submission()` 通过

---

## 4. 总分与评级

### 4.1 公式

```
workflow_score = Σ w_i · step_i   (仅对 non-None 步骤重归一；BUG-review #2)
                 ────────────────
                 Σ w_i
                   i ∈ active_steps

task_score = accuracy_judge         (open-ended 且启用 --enable-answer-judge)
           = 0.5·EM + 0.5·F1        (open-ended 无 judge；yes/no 走 strict)
           = exact-label accuracy   (multiple_choice)

overall = 0.5 · workflow_score + 0.5 · task_score
```

### 4.2 奖牌与评级

由 `medal_tier.py` 基于 `task_score` 分档 `tier ∈ {0, 1, 2}`。

```
rating = F   if not submission_format_valid
         or valid_outputs == 0
         or completion_rate < 0.5
       = A   if tier >= 2     (gold)
       = B   if tier >= 1     (silver/bronze, resolved)
       = C   otherwise        (ok but not resolved)
```

### 4.3 权重表

| 阶段 | 权重 | 判定源 |
|---|---|---|
| S1 Plan | 0.25 | `llm_judge.py`（可选） |
| S2 Setup | 0.15 | 三元 binary（env / model / smoke） |
| S3 Validate | **0.35** | `llm_judge.py` + postprocess cap |
| S4 Inference | 0.15 | completion + parse + guards |
| S5 Submit | 0.10 | valid + format |

---

## 5. Answer modes

任务 `config.yaml` 里的 `answer_mode` 决定打分公式。

| answer_mode | 任务 | task_score |
|---|---|---|
| `multiple_choice` | medxpertqa-mm / medframeqa | exact label (A–E) |
| `open_ended` | pathvqa / vqa-rad / slake | LLM judge（默认开）或 0.5·EM+0.5·F1 |

Yes/no 子类（gold ∈ {yes, no}）走 `yes_no_accuracy` 严格匹配，避免
"yes, the cyst wall ..." 因 F1 拿到虚高分。

---

## 6. LLM-as-judge（BUG-038）

**何时用**：`answer_mode=open_ended` 且传入 `--enable-answer-judge`（或设
`VQA_ANSWER_JUDGE=1`）。benchmark_runner 默认打开。

**模型**：`anthropic/claude-haiku-4.5` via OpenRouter；可由
`ANSWER_JUDGE_MODEL` / `ANSWER_JUDGE_BASE_URL` 覆盖。

**打分桶**：0.0 / 0.5 / 1.0（clinical incorrect / partial / equivalent）。

**缓存**：`<workspace>/answer_judge_cache.jsonl`，按
`sha256(qid∥gold∥pred∥model)` 去重。**Sweep 并行写入用 `fcntl.LOCK_EX`
独占锁**（review #1 修复），防止 JSONL 行交叉。

**Fallback**：缺 API key 或 backend 崩溃 → 退回 token-F1 / yes-no 启发式，
`judge_backend` 字段标为 `heuristic_fallback`；`run_eval.py` 会 stderr
WARNING（review #5 修复），避免静默退化为启发式打分。

---

## 7. 隐私合约

**强制**：S1–S4 期间 agent 只能读 `public_dir` 下的 `question.json`，
**禁止**任何 `answer*` / `reasoning_chain` / `correct_answer` /
`reference_answer` 字段。Gold 只在 S5 scoring 阶段从 `<gt_dir>/<qid>/answer.json`
加载。

`stage_hf_datasets.py` 负责拆分：`pub_root/<qid>/question.json` 只含
`question_id / dataset / question / image_paths / split / medical_task /
body_system / question_type`；`prv_root/<qid>/answer.json` 含
`answer_label / answer_text`。测试 `test_placeholder_detection.py` 等覆盖
staging 隐私合约。

---

## 8. Failure codes

由 `failure_classifier.py` 输出 `primary_failure`：

| 代码 | 含义 | 触发条件 |
|---|---|---|
| E1 | Hallucination | 回答与图像 / 问题明显不符（语义） |
| E2 | Resource error | smoke_forward 失败 / 环境未就绪 |
| E3 | Logic error | postprocess 不可 import / calibration 畸形 |
| E4 | Code error | execute_code 抛异常未恢复 |
| E5 | Format error | `model_call=False` 或 placeholder/empty 输出 |
| **E8** | S3 artefacts never written | smoke 过了但 pp.py + calibration.json 全缺 + 零 answer.json（BUG-044） |

---

## 9. 沙箱 / 隔离

Agent 视角的文件系统：

- 只读：`/data/public/`
- 读写：`/workspace/run_<id>/`
- 屏蔽：`/data/private/`、`eval_vqa/runs/`、其他 run 目录、harness 源码

Docker 隔离（推荐用于并行 sweep）：

```bash
bash eval_vqa/run_vqa_docker.sh
# 或
python eval_vqa/docker/orchestrator.py --agent ... --gpu-id N
```

`/dev/shm` 建议 ≥ 8 GiB（LLaVA-Med 7B mmap 多 worker 并发加载）。
Bus error 典型症状：`inference_mode=empty` + S2/S3 中途挂死（CLAUDE.md 已记）。

---

## 10. 开发与测试

```bash
cd eval_vqa
python -m pytest tests/ -v
```

当前 **103 tests passing**（review round 后）。覆盖面：

- `test_scoring.py` / `test_workflow_score_renorm.py` — 聚合 / 重归一化
- `test_answer_judge.py` — LLM judge + 并发缓存
- `test_tool_call_recovery.py` — BUG-043/040 XML 降级 + prose 防误检
- `test_failure_classifier_e8.py` — E8 S3 停滞
- `test_placeholder_detection.py` — 隐私 / 占位符
- `test_sweep_preflight.py` — `/dev/shm` 预检
- `test_prompt_open_ended_guidance.py` — open-ended 短答案契约注入
- `test_workspace_substitution.py` — workspace 路径替换（BUG-046）
- `test_length_finish_rate.py` — finish_reason=length 诊断（BUG-045）

---

## 11. 最近修复（2026-04 review round）

| # | 问题 | 修复 |
|---|---|---|
| 1 | `answer_judge` 缓存并发 race | `fcntl.LOCK_EX` + `fsync`，每行原子写入 |
| 2 | S1/S3 None 步骤静默零化 workflow | `compute_workflow_score` 按 active steps 重归一 |
| 3 | `tool_call_recovery` 对 prose 误检 | 要求 wrapper sentinel 或行首锚；prose 中 `<invoke>` 拒收 |
| 4 | `/dev/shm` 无预检 | `sweep.preflight_shm()` + `--allow-low-shm` |
| 5 | Judge heuristic fallback 静默 | `judge_fallback_count` 字段 + `run_eval` stderr WARNING |
| 6 | Turn-15 S3 预警仅查 pp.py | 同时查 `s3_calibration.json`，预警明示缺哪个 |
| 7 | Open-ended 短答案契约缺失 | `build_tier_system_prompt` 对 `answer_mode=open_ended` 注入 ≤5-word 契约 |

历史 bug ticket：`bug_issues/BUG-038` … `BUG-047`。改代码前请先扫相关 ticket。

---

## 12. 备注

- **不要**手工把 seg 的 agent_config.yaml 跟 VQA 的混用；VQA 用
  `nvda-<agent>` 形式的 provider-qualified key。
- **Tier 只影响 prompt**，不影响打分公式。lite / standard 共享
  `_WEIGHTS`。
- `old_vqa/` 和 `vqa_hard/` 是早期版本快照，**只读参考**，不要扩展。
- 所有运行产物落入 `runs/<experimenter>/...`；其他 experimenter 的目录不
  要读也不要 stage 进 git。
