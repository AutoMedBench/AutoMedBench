Calibrate postprocess + gold on **exactly 15 public samples** before S4.
Reuse the `run_llava_med(...)` helper you defined in S2 (from
`prompts/common/llava_med_skill.md`). Do NOT re-load the model here.

Skill — Loop 15 PathVQA samples and build calibration (examples only —
use any approach that works):

```python
import os, glob, json
from answer_postprocess import postprocess  # the module you are about to write

WORKSPACE = os.environ["WORKSPACE_DIR"]
samples = sorted(glob.glob("/data/public/*/question.json"))[:15]

records = []
for sp in samples:
    with open(sp) as f:
        q = json.load(f)
    qid = q["question_id"]
    img = os.path.join(os.path.dirname(sp), q["image_paths"][0])
    raw = run_llava_med(q.get("question", ""), img, max_new_tokens=128)

    # Gold lookup from the PUBLIC manifest only. For PathVQA, gold is in
    # `reference_answer` or `reasoning_chain.answer`. Never read from the
    # private /data/private tree.
    gold = (q.get("reference_answer") or "").strip()
    if not gold:
        chain = q.get("reasoning_chain") or {}
        gold = (chain.get("answer") or "").strip()

    # Drop samples with no retrievable gold — do NOT pad with "unknown".
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

# Top up if you dropped some — keep iterating over the next unused samples
# until len(records) >= 15. Report the hit rate to stderr.
with open(os.path.join(WORKSPACE, "s3_calibration.json"), "w") as f:
    json.dump(records, f, indent=2)
print("hit_rate:", round(sum(r["hit"] for r in records) / max(len(records), 1), 3))
```

Skill — `${WORKSPACE_DIR}/answer_postprocess.py` template (open-ended, PathVQA):

```python
"""Deterministic raw → short-answer mapper for PathVQA.

Must be importable with no side effects. Never raises. Always returns str.
"""
from __future__ import annotations
import re

_YES = {"yes", "y", "true", "correct"}
_NO = {"no", "n", "false", "incorrect"}
_YES_PREFIX = re.compile(r"^\s*(yes|yeah|yep|correct|true)\b", re.I)
_NO_PREFIX = re.compile(r"^\s*(no|nope|false|incorrect)\b", re.I)


def postprocess(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    text = raw.strip().strip('"').strip("'").rstrip(".")
    if not text:
        return ""
    low = text.lower()
    # Binary collapse — PathVQA has many yes/no items.
    if low in _YES:
        return "yes"
    if low in _NO:
        return "no"
    if _YES_PREFIX.match(text):
        return "yes"
    if _NO_PREFIX.match(text):
        return "no"
    # Short-phrase: take the first clause, drop trailing punctuation,
    # lowercase. Sentences longer than ~8 words are usually decode bleed;
    # keep the first phrase before the first comma/period.
    first = re.split(r"[,.;!?]", text, maxsplit=1)[0].strip().lower()
    return first or text.lower()
```

Checklist before S4:

- Verify `from answer_postprocess import postprocess` works in a fresh Python
  (no top-level downloads / no GPU imports at module level).
- `postprocess("no, the upper dermis shows only mild edema.")` must return
  `"no"`.
- `s3_calibration.json` has **>= 15 records** with real `gold_answer` pulled
  from `reference_answer` or `reasoning_chain.answer`. Drop, don't fabricate.
- `raw_model_output` length ≥ 5 chars; does not start with punctuation. If it
  does, your S2 decode dropped the prefix — fix the `output_ids` slicing
  before S4.
- If calibration `hit_rate < 0.20`, iterate on `postprocess` (e.g. strip
  answer prefixes like `"the answer is "`) before running S4 at scale.
