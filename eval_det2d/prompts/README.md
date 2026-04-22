# Prompt Templates — S1 through S5

These are the **generic** prompt templates for the 2D detection benchmark.
They use `{placeholders}` filled at runtime from the task config.

Task-specific **skills** (code examples) live in `eval_det2d/<task-id>/` folders,
NOT here. These templates define the S1-S5 stage instructions only.

```
prompts/
  common/
    preamble.md           Sandbox rules, task description, data paths, output format
    env_lite.md           Environment block for Lite tier
    env_standard.md       Environment block for Standard tier
    env_pro.md            Environment block for Pro tier
    important_lite.md     Important notes for Lite
    important_standard.md Important notes for Standard
    important_pro.md      Important notes for Pro
    kickoff.md            First user message per tier

  s1_plan/
    lite.md               "Your detector is {model_architecture}..."
    standard.md           "Explore these detector families: {model_range}..."
    pro.md                "Head-to-head competition, find the best detector..."

  s2_setup/
    lite.md               "pip install -r, download, load on GPU"
    standard_pro.md       "Create venv, download, load, check compatibility"

  s3_validate/
    lite_standard.md      "Run ONE image, GPU mandatory, check JSON + box validity"
    pro.md                "Run ONE image, use available resources"

  s4_inference/
    lite_standard.md      "Run ALL patients, GPU, maximize throughput"
    pro.md                "Run ALL patients, full resources"

  s5_submit/
    all.md                "Verify prediction.json files and call submit_results"
```

## How prompts are assembled

```
Preamble
+ Environment (per tier)
+ "## Workflow (S1 -> S5)"
+ S1 instructions (per tier) + S1 skill from task folder (lite/standard only)
+ S2 instructions (per tier) + S2 skill from task folder (lite only)
+ S3 instructions (per tier) + S3 skill from task folder (lite/standard only)
+ S4 instructions (per tier)
+ S5 instructions
+ Important notes (per tier)
```

Pro tier: no skills loaded from task folder.
