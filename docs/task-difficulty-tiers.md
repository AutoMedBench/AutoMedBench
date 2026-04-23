# Task Difficulty Tiers

Each task ships in 2–3 tiers, controlled by how much scaffolding the agent receives. Same data, same metrics — only the prompt changes.

| Tier | Model information | Dependencies | S1 / S2 / S3 skill hints | What it measures |
|---|---|---|---|---|
| **Lite** | Exact SOTA model named (e.g. "use nnU-Net KiTS19") | `requirements.txt` provided | S1 + S2 + S3 skills all provided | Execution competence: can the agent follow a recipe? |
| **Standard** | 2–5 candidate model families listed | Agent figures out pinning | S1 skill only | Judgment under constrained choice: can it pick wisely? |
| **Pro** | Only medical background text | Agent figures everything out | No skills | Open-research capability: can it design from scratch? |

## Why three tiers

An agent that scores well on **Lite** but poorly on **Pro** is a capable executor but not a researcher. An agent that scores equally across tiers is genuinely autonomous. This split keeps the benchmark useful as frontier models improve — as Lite saturates, Standard and Pro remain informative.

## Tier availability

| Domain | Lite | Standard | Pro |
|---|:---:|:---:|:---:|
| `eval_seg` | yes | yes | (planned) |
| `eval_image_enhancement` | yes | yes | — |
| `eval_report_gen` | yes | yes | yes |
| `eval_vqa` | yes | yes | — |
| `eval_det2d` (beta) | yes | yes | yes |
