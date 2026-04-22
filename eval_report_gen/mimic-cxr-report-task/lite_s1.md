Skill — How to download checkpoints / inspect the model / run inference (examples only — use any approach that works):
```python
# Example: set up MLRG from the official repo and official HuggingFace checkpoint
#
# Repo:
#   https://github.com/mk-runner/MLRG
#
# Official checkpoint:
#   https://huggingface.co/MK-runner/MLRG
#
# The official README says to:
# - create a Python 3.9 environment
# - install with: pip install -r requirements.txt
# - place all checkpoints in one local checkpoint zoo directory
# - set --ckpt_zoo_dir <your_checkpoint_dir> in script/MIMIC-CXR/*.sh

# 1) Clone the official repo
from subprocess import run
run(["git", "clone", "https://github.com/mk-runner/MLRG.git", "MLRG"], check=False)

# 2) Download the official MLRG checkpoint from HuggingFace
# Note: the HuggingFace model page may be gated and require login/approval.
from huggingface_hub import snapshot_download

ckpt_root = "/path/to/checkpoints"
snapshot_download(
    repo_id="MK-runner/MLRG",
    local_dir=ckpt_root + "/MK-runner_MLRG",
    local_dir_use_symlinks=False,
)

# 3) Inspect the repository structure
import os
repo_dir = "MLRG"
for root, dirs, files in os.walk(repo_dir):
    if root.count(os.sep) - repo_dir.count(os.sep) > 2:
        continue
    print(root)
    for f in files:
        print(" ", os.path.join(root, f))

# Look for:
# - README.md
# - requirements.txt
# - main_v0926_ablation_study.py
# - script/MIMIC-CXR/run_cxr_pt_v0906_fs.sh
# - script/MIMIC-CXR/run_cxr_ft_mlrg_v1011.sh
# - tools/dataset/datasets_v0818_ab.py

# 4) Inspect the official run scripts and checkpoint variables
targets = [
    "README.md",
    "requirements.txt",
    "script/MIMIC-CXR/run_cxr_pt_v0906_fs.sh",
    "script/MIMIC-CXR/run_cxr_ft_mlrg_v1011.sh",
    "main_v0926_ablation_study.py",
]
for rel in targets:
    path = os.path.join(repo_dir, rel)
    if os.path.isfile(path):
        print(f"=== {rel} ===")
        with open(path, encoding="utf-8", errors="replace") as f:
            print(f.read()[:3000])

# Verify where these are configured:
# - --ckpt_zoo_dir
# - chexbert_path
# - bert_path
# - radgraph_path
# - rad_dino_path
# - cxr_bert_path
# - distilgpt2_path
# - cvt2distilgpt2_path

# 5) Inspect what text target the model uses
dataset_code = os.path.join(repo_dir, "tools/dataset/datasets_v0818_ab.py")
with open(dataset_code, encoding="utf-8", errors="replace") as f:
    print(f.read()[:3000])

# Check whether the code uses:
# - findings
# - findings_factual_serialization
# - prior_study
# - multi-view image_path inputs

# 6) Adaptation goal for this benchmark
# Benchmark contract:
# - input: public/<case_id>/images/*.jpg and manifest.json
# - output: {output_dir}/agent_outputs/<case_id>/report.txt
# - evaluation reference: findings-only text
#
# Your S1 goal is to record:
# - how to obtain the official checkpoint
# - how to point the repo to the checkpoint zoo
# - how to invoke the model's inference path
# - how to adapt its input/output contract to this benchmark
```
