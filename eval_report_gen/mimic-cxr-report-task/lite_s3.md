Skill — Environment / resource validation before S3 and S4 (examples only — use any approach that works):
```python
import os

repo_dir = "{output_dir}/MLRG"
env_dir = "{output_dir}/env"
ckpt_root = "{output_dir}/checkpoints"
public_dir = "{data_dir}"

print("repo exists:", os.path.isdir(repo_dir))
print("env exists:", os.path.isdir(env_dir))
print("checkpoints exist:", os.path.isdir(ckpt_root))
print("public data exists:", os.path.isdir(public_dir))

# Verify that the benchmark staged data is present
case_ids = sorted([d for d in os.listdir(public_dir) if os.path.isdir(os.path.join(public_dir, d))])
print("cases:", case_ids[:10])
assert len(case_ids) > 0, "No staged cases found"

# Verify that a pilot case has images and a manifest
case_dir = os.path.join(public_dir, case_ids[0])
print("pilot case:", case_ids[0])
print("images:", os.listdir(os.path.join(case_dir, "images")))
print(open(os.path.join(case_dir, "manifest.json")).read())
```

Skill — How to validate on one study (examples only — use any approach that works):
```python
import os
import json

public_dir = "{data_dir}"
output_dir = "{output_dir}"

# Pick the first staged case for smoke testing
case_ids = sorted([d for d in os.listdir(public_dir) if os.path.isdir(os.path.join(public_dir, d))])
case_id = case_ids[0]
case_dir = os.path.join(public_dir, case_id)

with open(os.path.join(case_dir, "manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)

print("case_id:", case_id)
print("image_count:", manifest["image_count"])
print("image_files:", manifest["image_files"])

# Run your model-specific single-case inference here
# ... your inference code here ...

# Verify the saved report
report_path = os.path.join(output_dir, "agent_outputs", case_id, "report.txt")
assert os.path.isfile(report_path), f"Missing report: {report_path}"

with open(report_path, encoding="utf-8", errors="replace") as f:
    report = f.read()

print(report)
print("char_count:", len(report))
print("non_empty:", len(report.strip()) > 0)

# Findings-only benchmark checks
has_findings = "FINDINGS:" in report
has_impression = "IMPRESSION:" in report
print("has_findings:", has_findings)
print("has_impression:", has_impression)

assert len(report.strip()) > 0, "Report is empty"

# Recommended validation checks:
# - the env versions match the selected method's documented or declared versions
# - all views listed in manifest were used
# - report is non-empty
# - report is findings-oriented and clinically grounded
# - no obvious hallucinated acute pathology
# - support devices / chronic findings are only mentioned if visible
```
