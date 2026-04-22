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
print("image_paths:", q.get("image_paths"))
# SLAKE (English test split): single radiology image (CT / MRI / X-ray).
# question_type is "open" (organ / modality / plane) or "binary" (yes/no).
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
   conversation template; `DEFAULT_IMAGE_TOKEN` in user turn; decode only
   newly-generated tokens.
2. Data — SLAKE English test split at `/data/public/<qid>/question.json` with
   single radiology image. Canonical short-phrase answers (organ names,
   modality, anatomical plane) plus binary yes/no.
3. Answer schema — `predicted_label` stays `""`. Binary samples collapse to
   exactly `yes` or `no`. Open-ended short phrases lowercased, leading
   `"the answer is "` stripped, trailing punctuation removed.
4. Smoke plan — 1–10 samples through S2 `run_llava_med(...)`;
   `smoke_forward.json` must record a real non-empty decode.
5. Helpers — `inspect_image` only when decode looks wrong; `submit_answer`
   once per qid in S4. No `public_medical_search` needed.
