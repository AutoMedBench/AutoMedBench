Skill — How to set up the environment (examples only — use any approach that works):
```bash
# Create a clean environment
python3.9 -m venv {output_dir}/env
source {output_dir}/env/bin/activate

# Clone the official repo
git clone https://github.com/mk-runner/MLRG.git {output_dir}/MLRG

# Install base dependencies from the official repo
pip install -r {output_dir}/MLRG/requirements.txt

# Install additional packages if needed
pip install huggingface_hub
```

```python
# After choosing a method, follow this order:
# 1. official inference entrypoint
# 2. official config / checkpoint wiring
# 3. official environment / dependency versions used by that method
# 4. official output/export logic
# 5. only then a minimal benchmark adapter if the repo does not provide step 4
# Before writing install/run scripts, first detect which executables actually exist:
# - python
# - python3
# - python3.9 / python3.10 / python3.11 / python3.12
# - pip / pip3
# Do not assume `python` exists on the machine.
# Use a verified interpreter path in every script you generate.
# If the selected method needs a versioned environment that is missing, first try to
# create or recover that environment yourself using the interpreters that do exist;
# only treat the environment as a blocker after that recovery attempt fails.
# Do not reimplement the model pipeline if the repo already provides a runnable path.
# Limit adaptations to:
# - mapping benchmark inputs into the method's expected input structure
# - mapping method outputs into agent_outputs/<case_id>/report.txt
#
# If the official path is blocked (missing access, missing weights, broken dependencies),
# document that failure first, then fall back to the next-best method.
# Before running inference, inspect the selected method's README, notebook, generation_config.json,
# config.json, requirements.txt, or lockfile to recover the intended versions for transformers,
# torch, torchvision, tokenizers, and any remote-code dependencies.
# If the model config declares a transformers_version, prefer matching that version in a clean env
# before attempting code patches.
# If the official repo defines how to export findings, impression, or full reports,
# follow that method-specific contract rather than inventing a generic split rule.
# If the repo does not define an export helper, document that absence before writing one.

# Download the official MLRG checkpoint from HuggingFace
from huggingface_hub import snapshot_download

ckpt_root = "{output_dir}/checkpoints"
snapshot_download(
    repo_id="MK-runner/MLRG",
    local_dir=ckpt_root + "/MK-runner_MLRG",
    local_dir_use_symlinks=False,
)

# Inspect the official scripts and checkpoint wiring
import os

repo_dir = "{output_dir}/MLRG"
for exe in ["python", "python3", "python3.9", "python3.10", "python3.11", "python3.12", "pip", "pip3"]:
    print(exe, "->", os.popen(f"command -v {exe}").read().strip() or "MISSING")

targets = [
    "README.md",
    "script/MIMIC-CXR/run_cxr_pt_v0906_fs.sh",
    "script/MIMIC-CXR/run_cxr_ft_mlrg_v1011.sh",
    "main_v0926_ablation_study.py",
]
for rel in targets:
    path = os.path.join(repo_dir, rel)
    if os.path.isfile(path):
        print(f"=== {rel} ===")
        with open(path, encoding="utf-8", errors="replace") as f:
            print(f.read()[:2500])

# Verify where these are configured:
# - --ckpt_zoo_dir
# - chexbert_path
# - bert_path
# - radgraph_path
# - rad_dino_path
# - cxr_bert_path
# - distilgpt2_path
# - cvt2distilgpt2_path

# Adapt only the benchmark contract:
# - read staged images from public/<case_id>/images/
# - use manifest.json to enumerate all views for a study
# - save exactly one findings-style report to:
#     {output_dir}/agent_outputs/<case_id>/report.txt

# Prefer the repo's official scripts/configs/export path over a custom inference rewrite.
# If the method has an official section parser, decoder helper, or export utility,
# use that instead of hand-writing a findings/impression splitter.
# If you cannot use the official path, record the exact blocker in plan.md before deviating.

# Before full batch generation:
# - run a 1-case smoke test
# - inspect the saved report.txt
# - confirm the output is findings-oriented and non-empty
```
