You are given the exact starting model:

- `microsoft/llava-med-v1.5-mistral-7b` (LLaVA-Med 7B, open weights, `conv_templates["mistral_instruct"]`)

This model is FIXED for lite tier. Do not swap to another VLM.

Skill — Download the weights and inspect the staged data (examples only —
use any approach that works):

```python
import os, glob, json
from huggingface_hub import snapshot_download

WORKSPACE = os.environ["WORKSPACE_DIR"]
model_path = snapshot_download(
    "microsoft/llava-med-v1.5-mistral-7b",
    local_dir=os.path.join(WORKSPACE, "LLaVA-Med", "weights"),
)
print("weights at:", model_path)

sample = sorted(glob.glob("/data/public/*/question.json"))[0]
with open(sample) as f:
    q = json.load(f)
print("keys:", sorted(q.keys()))
print("question:", q.get("question"))
print("options:", q.get("options"))
print("valid_labels:", q.get("valid_labels"))
print("image_paths:", q.get("image_paths"))
# MedFrameQA: an ORDERED sequence of medical frames (multiple images per
# sample); MCQ with A-E options. `image_paths` is a list — show all frames
# to the model, in order.
```

Skill — Load the model via the LLaVA-Med helper:

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_8bit=False,
    load_4bit=False,
    device_map={"": 0},
)
model.eval()
print("loaded on:", model.device, "dtype:", model.dtype)
```

Write `${WORKSPACE_DIR}/plan/plan.md` covering:

1. Model — `microsoft/llava-med-v1.5-mistral-7b`; `mistral_instruct` conv
   template. For multi-frame input: one `DEFAULT_IMAGE_TOKEN` per frame in
   the user turn, in order; stack `process_images(...)` on all frames into
   one 4-D tensor before `.generate(...)`.
2. Data — MedFrameQA samples at `/data/public/<qid>/question.json`,
   `image_paths` is a list of frames in presentation order; `options` is a
   dict A–E; `valid_labels` constrains the answer letter.
3. Answer schema — MCQ: `predicted_label` is an uppercase letter in
   `valid_labels`; `predicted_answer` is the matching option text copied
   from `options[predicted_label]`.
4. Prompt plan — build `"Frame 1: <img>\nFrame 2: <img>\n...Question: ...\n
   Options:\nA) ...\n...\nAnswer with a single letter."`; parse first
   A–E letter from raw decode.
5. Smoke plan — 1–10 samples through S2; ensure all frames in one sample
   are fed before calling `.generate(...)`; record a real non-empty decode
   in `smoke_forward.json`.
6. Helpers — `inspect_image` when decoding is suspect; `submit_answer` once
   per qid in S4.
