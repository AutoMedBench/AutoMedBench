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
# MedXpertQA-MM: multi-image clinical MCQ with A-E options. `valid_labels`
# is the canonical A-E letter list; `options` is the dict label -> text.
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

1. Model — `microsoft/llava-med-v1.5-mistral-7b`; `mistral_instruct`
   conversation template; `DEFAULT_IMAGE_TOKEN` in user turn per image;
   for multi-image samples, put one image token per image or concatenate.
2. Data — MedXpertQA-MM samples at `/data/public/<qid>/question.json` with
   `image_paths` (1+ clinical images), `question`, `options` dict A–E,
   `valid_labels` list. Read-only at S1.
3. Answer schema — this is MCQ. `predicted_label` must be an uppercase
   letter in `valid_labels`; `predicted_answer` is the matching option text
   (copy from `options[predicted_label]`, not invented).
4. Prompt plan — build a prompt like `"Question: {q}\nOptions:\nA) ...\nB)
   ...\n...\nAnswer with a single letter."`; parse the first letter in A–E
   from the raw decode.
5. Smoke plan — 1–10 samples through S2; record a real non-empty decode in
   `smoke_forward.json` before S3.
6. Helpers — `inspect_image` only when decode looks wrong; `submit_answer`
   once per qid in S4.
