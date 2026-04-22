Calibrate postprocess + gold on **exactly 15 public samples**. Reuse the
`run_llava_med(...)` helper from S2. Do not re-load the model.

For MCQ, the postprocess extracts a single label in `valid_labels` from the
raw decode, then the scorer compares `predicted_label` to the gold letter.

Skill — Loop 15 MedXpertQA-MM samples and build calibration (examples only —
use any approach that works):

```python
import os, glob, json
from answer_postprocess import postprocess  # returns (label, answer_text)

WORKSPACE = os.environ["WORKSPACE_DIR"]
samples = sorted(glob.glob("/data/public/*/question.json"))[:15]

def build_mcq_prompt(q):
    opts = q.get("options") or {}
    lines = [f"Question: {q.get('question','')}", "Options:"]
    for letter in sorted(opts):
        lines.append(f"{letter}) {opts[letter]}")
    lines.append("Answer with a single letter.")
    return "\n".join(lines)

records = []
for sp in samples:
    with open(sp) as f:
        q = json.load(f)
    qid = q["question_id"]
    img = os.path.join(os.path.dirname(sp), q["image_paths"][0])
    raw = run_llava_med(build_mcq_prompt(q), img, max_new_tokens=64)

    gold_letter = (q.get("reference_answer") or "").strip().upper()
    if not gold_letter:
        gold_letter = ((q.get("reasoning_chain") or {}).get("answer") or "").strip().upper()
    valid = set((q.get("valid_labels") or ["A","B","C","D","E"]))
    if gold_letter not in valid:
        continue

    pred_label, pred_text = postprocess(raw, valid_labels=sorted(valid), options=q.get("options") or {})
    records.append({
        "question_id": qid,
        "raw_model_output": raw,
        "predicted_label": pred_label,
        "predicted_answer": pred_text,
        "gold_answer": gold_letter,
        "hit": pred_label == gold_letter,
    })

with open(os.path.join(WORKSPACE, "s3_calibration.json"), "w") as f:
    json.dump(records, f, indent=2)
print("hit_rate:", round(sum(r["hit"] for r in records) / max(len(records), 1), 3))
```

Skill — `${WORKSPACE_DIR}/answer_postprocess.py` template (MCQ, MedXpertQA-MM):

```python
"""Deterministic raw → (label, option_text) mapper for MedXpertQA-MM.

The scorer imports `postprocess(raw)` and probes it with a plain string
(no valid_labels) — so the callable must also support that signature.
It returns the raw short answer string (option text) when called with one
arg, and a (label, text) tuple when called with `valid_labels=`.

Must import with no side effects. Must never raise. Always returns str-like.
"""
from __future__ import annotations
import re
from typing import Iterable, Mapping, Tuple, Union

_DEFAULT_LABELS = ["A", "B", "C", "D", "E"]
_LEAD = re.compile(r"^(the\s+answer\s+is|answer\s*:|final\s+answer\s*:)\s*", re.I)


def _extract_letter(text: str, valid_labels: Iterable[str]) -> str:
    labels = [l.upper() for l in valid_labels]
    # Prefer "Answer: X" / "(X)" / "X)" patterns, then any standalone letter.
    m = re.search(r"\b([A-E])\b", text.upper())
    if m and m.group(1) in labels:
        return m.group(1)
    return ""


def postprocess(
    raw: str,
    valid_labels: Iterable[str] | None = None,
    options: Mapping[str, str] | None = None,
) -> Union[str, Tuple[str, str]]:
    if not isinstance(raw, str):
        raw = ""
    text = _LEAD.sub("", raw.strip()).strip('"').strip("'")
    labels = list(valid_labels) if valid_labels else _DEFAULT_LABELS
    letter = _extract_letter(text, labels)
    if valid_labels is None and options is None:
        # Scorer probe path — return short answer text only.
        return letter or text.split("\n", 1)[0].strip().lower()
    opts = options or {}
    return letter, (opts.get(letter, "") if letter else "").strip()
```

Checklist before S4:

- `postprocess("The answer is C.")` → returns `"C"` (string mode) /
  `("C", <option_text>)` (tuple mode).
- `s3_calibration.json` ≥ 15 records with real gold letter from
  `reference_answer` / `reasoning_chain.answer`. Drop, don't fabricate.
- Every `predicted_label` in calibration is in `valid_labels` (uppercase
  A–E). Empty label = parse miss — iterate on the regex before S4.
- S4 must write `predicted_answer = options[predicted_label]` (not raw text).
