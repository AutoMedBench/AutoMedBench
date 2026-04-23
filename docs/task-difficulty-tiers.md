# Task Difficulty Tiers

Same data, same metric — different scaffolding. Each row below is something that *changes* across tiers.

|  | **Lite** | **Standard** | **Pro** |
|---|:---|:---|:---|
|  | *follows a recipe* | *chooses within bounds* | *discovers and competes* |
| **Autonomy level** | Low &nbsp; ● | Medium &nbsp; ● ● | High &nbsp; ● ● ● |
| **Model information given to agent** | Exact SOTA model named (e.g. "use `nnU-Net KiTS19`") | 2–5 candidate model families listed (e.g. "MMDetection or YOLO family") | Medical background text only — agent researches candidates itself |
| **Dependencies** | `requirements.txt` provided and pinned | Agent resolves and pins versions | Agent resolves and pins versions |
| **S1–S3 skill hints in prompt** | All three (Plan · Setup · Validate) provided | S1 Plan only | None |
| **Required plan artifacts** | `plan.md` | `plan.md` + `plan.png` | `plan.md` + `plan.png` + summary plots |
| **Research requirement** | — *(model is given)* | Compare ≥ 3 candidates in the plan | Justify a method from scratch |
| **S1 rubric auto-passes** | `s1d` research · `s1e` plan.png · `s1f` diagram quality | — | — |
| **What it tests** | Can the agent execute a known recipe cleanly? | Can it choose wisely from a short list? | Can it design and validate from only clinical context? |

## Tier availability

| Domain | Lite | Standard | Pro |
|---|:---:|:---:|:---:|
| `eval_seg` | ✓ | ✓ | planned |
| `eval_image_enhancement` | ✓ | ✓ | — |
| `eval_vqa` | ✓ | ✓ | — |
| `eval_report_gen` | ✓ | ✓ | ✓ |
| `eval_det2d` (beta) | ✓ | ✓ | ✓ |

Same agent on the same task across tiers is the sharpest measure of how much of the "research pipeline" the agent can actually do unassisted. Scores typically drop by 0.15–0.35 on Agentic as you move Lite → Standard → Pro.
