Skill — How to research and compare open-source report-generation models (examples only — use any approach that works):
```python
# Search HuggingFace or GitHub for candidate report-generation models
import requests

hf_resp = requests.get(
    "https://huggingface.co/api/models",
    params={"search": "chest x-ray report generation", "limit": 10},
    timeout=30,
)
for model in hf_resp.json():
    print(model["modelId"], model.get("downloads"))

# Inspect a model card / repo README
readme = requests.get(
    "https://raw.githubusercontent.com/mk-runner/MLRG/main/README.md",
    timeout=30,
)
print(readme.text[:2000])

# Download a public checkpoint snapshot for inspection
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="aehrc/cxrmate",
    local_dir="{output_dir}/candidate_model",
    ignore_patterns=["*.pt", "*.bin.index.json"],
)

# Check whether the method has an official inference entrypoint
import os
for root, _, files in os.walk("{output_dir}/candidate_repo"):
    for name in files:
        if name.endswith((".py", ".sh", ".ipynb")) and "infer" in name.lower():
            print(os.path.join(root, name))

# Also inspect whether the method documents an official report-export or section-extraction path.
# Examples:
# - helper functions that split findings / impression
# - notebook cells that decode and post-process model output
# - repo utilities that convert raw generations into final reports
# If such a path exists, the benchmark should reuse it rather than invent a generic splitter.
# If no such path exists, note that explicitly in plan.md before proposing a custom adapter.
#
# Before committing to a setup plan, explicitly detect the local environment:
# - which Python executables exist (`python`, `python3`, `python3.9`, `python3.10`, etc.)
# - which pip executables exist
# - whether the selected method's requested Python version is already present
# Do not assume `python` exists.
# The setup plan should use a verified interpreter path, and if the official environment
# is missing, it should attempt to recover that environment before declaring a blocker.
```

```markdown
# Comparison table template for plan.md:
| Model | Multi-view | Longitudinal | Official checkpoint | Official inference path | Official export logic | If export missing, fallback plan | Public evidence | Notes |
|-------|------------|--------------|---------------------|-------------------------|-----------------|-------|
| ... | ... | ... | ... | ... | ... | ... |
```
