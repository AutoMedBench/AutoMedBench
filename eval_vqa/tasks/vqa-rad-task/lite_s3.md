Calibrate postprocess + gold on **exactly 15 public samples**. Reuse the
`run_llava_med(...)` helper from S2. Do not re-load the model.

Skill — Loop 15 VQA-RAD samples and build calibration (examples only —
use any approach that works):

```python
import os, glob, json
from answer_postprocess import postprocess

WORKSPACE = os.environ["WORKSPACE_DIR"]
samples = sorted(glob.glob("/data/public/*/question.json"))[:15]

records = []
for sp in samples:
    with open(sp) as f:
        q = json.load(f)
    qid = q["question_id"]
    img = os.path.join(os.path.dirname(sp), q["image_paths"][0])
    raw = run_llava_med(q.get("question", ""), img, max_new_tokens=128)

    gold = (q.get("reference_answer") or "").strip()
    if not gold:
        gold = ((q.get("reasoning_chain") or {}).get("answer") or "").strip()
    if not gold or gold.lower() in {"unknown", "n/a", "na", "none", "null", "?", "-"}:
        continue

    pred = postprocess(raw)
    records.append({
        "question_id": qid,
        "raw_model_output": raw,
        "predicted_answer": pred,
        "gold_answer": gold,
        "hit": pred.strip().lower().rstrip(".,;:!?") == gold.strip().lower().rstrip(".,;:!?"),
    })

with open(os.path.join(WORKSPACE, "s3_calibration.json"), "w") as f:
    json.dump(records, f, indent=2)
print("hit_rate:", round(sum(r["hit"] for r in records) / max(len(records), 1), 3))
```

Skill — `${WORKSPACE_DIR}/answer_postprocess.py` template (VQA-RAD, open-ended + binary):

```python
"""Deterministic raw → short-answer mapper for VQA-RAD.

Must import with no side effects. Must never raise. Always returns str.
VQA-RAD mixes binary (yes/no) and open-ended (anatomy, imaging modality,
finding). Binary collapse rules are strict — `yes_no_accuracy` only
credits exact `yes` / `no`.
"""
from __future__ import annotations
import re

_YES_PREFIX = re.compile(r"^\s*(yes|yeah|yep|correct|true|positive)\b", re.I)
_NO_PREFIX = re.compile(r"^\s*(no|nope|false|incorrect|negative)\b", re.I)
_LEAD = re.compile(r"^(the\s+answer\s+is|answer\s*:|final\s+answer\s*:)\s*", re.I)


def postprocess(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    text = _LEAD.sub("", raw.strip()).strip('"').strip("'").rstrip(".")
    if not text:
        return ""
    if _YES_PREFIX.match(text):
        return "yes"
    if _NO_PREFIX.match(text):
        return "no"
    first = re.split(r"[,.;!?]", text, maxsplit=1)[0].strip().lower()
    return first or text.lower()
```

Checklist before S4:

- `postprocess("yes, there is pleural effusion on the right.")` → `"yes"`.
- `postprocess("the answer is pneumothorax.")` → `"pneumothorax"`.
- `s3_calibration.json` has **>= 15 records** with real gold from
  `reference_answer` / `reasoning_chain.answer`. Drop samples with missing
  gold rather than fabricating.
- `raw_model_output` must be the full decoded text (≥ 5 chars, no
  punctuation prefix). If truncated, fix S2 decode slicing before S4.
- If `hit_rate < 0.20`, iterate on the binary-prefix rules before S4.
