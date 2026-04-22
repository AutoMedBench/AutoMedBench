Skill — Local model setup and readiness check (MANDATORY for S3 and S4, examples only — use any approach that works):
```python
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# If the selected method is a local checkpointed model, prefer GPU inference.
if os.path.exists("{output_dir}/candidate_model") or os.path.exists("{output_dir}/cxrmate_model"):
    assert device.type == "cuda", "GPU not available — check CUDA_VISIBLE_DEVICES"

# Verify that the official model assets you selected in S1 are present.
for path in [
    "{output_dir}/candidate_model",
    "{output_dir}/candidate_repo",
    "{output_dir}/cxrmate_model",
]:
    if os.path.exists(path):
        print("FOUND", path)
```

```bash
# Example GPU setup for local report-generation inference
export CUDA_VISIBLE_DEVICES=0
python3 inference.py CXR0001

# If the chosen repo provides an official script or notebook entrypoint,
# validate that path first before falling back to a wrapper.
# If the chosen repo provides an official output/export helper,
# validate that helper second, before inventing your own section split.
# Before any pilot run, verify the exact interpreter you plan to use exists.
# Prefer an explicit verified path such as `python3` or `/abs/path/to/python`
# over an assumed `python` executable.
# If the selected method requires a missing Python version, attempt to recover
# that environment first and document the recovery attempt.
```

Skill — How to validate on one study (examples only — use any approach that works):
```python
import json
import os

public_dir = "{data_dir}"
case_id = sorted(
    d for d in os.listdir(public_dir)
    if os.path.isdir(os.path.join(public_dir, d))
)[0]

manifest_path = os.path.join(public_dir, case_id, "manifest.json")
with open(manifest_path) as f:
    manifest = json.load(f)

image_files = manifest["image_files"]
print(f"Pilot case: {case_id}")
print(f"Views: {image_files}")
assert len(image_files) >= 1

# Run your selected method on exactly this study first.
# ... your inference code here ...

# Validate that the saved report text comes from the selected method's official
# output/export semantics. Do not silently replace the model's official section
# logic with your own generic findings splitter unless the repo gives no official path.
# If you had to write a custom adapter, note that the run is an adaptation rather than
# a faithful reproduction of the method's export behavior.

report_path = os.path.join("{output_dir}", "agent_outputs", case_id, "report.txt")
assert os.path.isfile(report_path), f"Missing {report_path}"

with open(report_path, "r", encoding="utf-8") as f:
    report = f.read().strip()

print(report)
assert len(report) >= 40, f"Report too short: {len(report)} chars"
assert sum(ch.isalpha() for ch in report) >= 20, "Report lacks enough alphabetic content"

# Findings-only benchmark: plain findings text is valid.
# If the model emits FINDINGS/IMPRESSION sections, confirm the findings part is sensible.
print(f"Chars: {len(report)}, Words: {len(report.split())}")
print("Pilot validation: OK")
```
