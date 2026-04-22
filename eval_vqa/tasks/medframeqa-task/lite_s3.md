Calibrate postprocess + gold on **exactly 15 public samples**. Reuse the
multi-frame `run_llava_med(...)` variant from S2. Do not re-load the model.

For MCQ, postprocess extracts a single letter in `valid_labels`; S4 then
maps that letter to `options[letter]` for `predicted_answer`.

Skill — Loop 15 MedFrameQA samples with multi-frame input (examples only —
use any approach that works):

```python
import os, glob, json
from PIL import Image
from answer_postprocess import postprocess

WORKSPACE = os.environ["WORKSPACE_DIR"]
samples = sorted(glob.glob("/data/public/*/question.json"))[:15]

def build_mcq_prompt(q, n_frames):
    opts = q.get("options") or {}
    lines = [f"You are shown {n_frames} ordered medical frames."]
    for i in range(n_frames):
        lines.append(f"Frame {i+1}: <image>")
    lines.append(f"Question: {q.get('question','')}")
    lines.append("Options:")
    for letter in sorted(opts):
        lines.append(f"{letter}) {opts[letter]}")
    lines.append("Answer with a single letter.")
    return "\n".join(lines)

records = []
for sp in samples:
    with open(sp) as f:
        q = json.load(f)
    qid = q["question_id"]
    frames = [os.path.join(os.path.dirname(sp), p) for p in q["image_paths"]]
    raw = run_llava_med_multi(build_mcq_prompt(q, len(frames)), frames, max_new_tokens=64)

    gold_letter = (q.get("reference_answer") or "").strip().upper()
    if not gold_letter:
        gold_letter = ((q.get("reasoning_chain") or {}).get("answer") or "").strip().upper()
    valid = set(q.get("valid_labels") or ["A","B","C","D","E"])
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

Skill — `${WORKSPACE_DIR}/answer_postprocess.py` template (MCQ, MedFrameQA):

```python
"""Deterministic raw → (label, option_text) mapper for MedFrameQA.

Supports both the scorer probe (single-arg string in, string out) and the
S3/S4 tuple path (valid_labels + options provided). Must import with no
side effects. Must never raise.
"""
from __future__ import annotations
import re
from typing import Iterable, Mapping, Tuple, Union

_DEFAULT_LABELS = ["A", "B", "C", "D", "E"]
_LEAD = re.compile(r"^(the\s+answer\s+is|answer\s*:|final\s+answer\s*:)\s*", re.I)


def _extract_letter(text: str, labels: Iterable[str]) -> str:
    up = text.upper()
    allowed = {l.upper() for l in labels}
    m = re.search(r"\b([A-E])\b", up)
    if m and m.group(1) in allowed:
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
        return letter or text.split("\n", 1)[0].strip().lower()
    opts = options or {}
    return letter, (opts.get(letter, "") if letter else "").strip()
```

Checklist before S4:

- `postprocess("Answer: B")` returns `"B"` (string mode) or
  `("B", <option>)` (tuple mode).
- `s3_calibration.json` ≥ 15 records with real gold letter. Drop samples
  where gold is not in `valid_labels` rather than fabricating.
- All frames in each sample were passed to the model (inspect your
  `run_llava_med_multi` — number of image tokens must match
  `len(image_paths)`).
- Every calibration `predicted_label` is uppercase A–E. Empty label →
  parser miss; iterate on the regex before S4.
