You are given the exact starting model:

- `microsoft/llava-med-v1.5-mistral-7b` (LLaVA-Med 7B, open weights, `conv_templates["mistral_instruct"]`)

This model is FIXED for lite tier. Do not swap to another VLM.

Skill — Download the weights and inspect the staged data (examples only —
use any approach that works; the snippets below are the vendor-documented path):

```python
# 1. Download LLaVA-Med weights once into the workspace.
import os
from huggingface_hub import snapshot_download

WORKSPACE = os.environ["WORKSPACE_DIR"]
model_path = snapshot_download(
    "microsoft/llava-med-v1.5-mistral-7b",
    local_dir=os.path.join(WORKSPACE, "LLaVA-Med", "weights"),
)
print("weights at:", model_path)

# 2. Inspect the staged PathVQA layout (public split, S1 is read-only).
import glob, json
sample = sorted(glob.glob("/data/public/*/question.json"))[0]
print("sample path:", sample)
with open(sample) as f:
    q = json.load(f)
print("keys:", sorted(q.keys()))
print("question:", q.get("question"))
print("image_paths:", q.get("image_paths"))
# Expected keys: question_id, question, image_paths (single pathology image),
# question_type ("open"|"binary"), valid_labels (empty for open-ended).
```

Skill — Load the model with LLaVA-Med's helper so S2/S3/S4 reuse it:

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),  # "llava-med-v1.5-mistral-7b"
    load_8bit=False,
    load_4bit=False,
    device_map={"": 0},
)
model.eval()
print("loaded on:", model.device, "dtype:", model.dtype)
```

Write `${WORKSPACE_DIR}/plan/plan.md` covering:

1. Model — `microsoft/llava-med-v1.5-mistral-7b`; conversation template =
   `mistral_instruct`; image token = `DEFAULT_IMAGE_TOKEN` injected into the
   user turn; decode only newly-generated tokens
   (`output_ids[:, input_ids.shape[-1]:]`).
2. Data — PathVQA samples under `/data/public/<qid>/question.json` + single
   pathology `image_paths[0]`. `question_type` is either `open` or `binary`
   (yes/no); no `valid_labels`.
3. Answer schema — `predicted_label` stays `""` (open-ended); `predicted_answer`
   is a canonical short phrase (lowercase, trimmed). Binary samples collapse to
   exactly `yes` or `no`.
4. Smoke plan — run `run_llava_med(...)` on 1–10 samples in S2 and write
   `smoke_forward.json` with a real non-empty decode before S3.
5. Helpers — `inspect_image` to double-check a pathology tile before S3 if
   decode looks wrong; `submit_answer` once per qid in S4. `public_medical_search`
   is not needed for a local VLM pipeline.
