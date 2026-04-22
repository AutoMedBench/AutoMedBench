Skill — End-to-end LLaVA-Med 7B setup and smoke forward (examples only —
use any approach that works; the code below is the vendor-documented path).

This skill is mandatory for lite tier: the fixed model is
`microsoft/llava-med-v1.5-mistral-7b` and its `.generate(...)` returns an
empty string when called on a bare question. Follow the `conv_templates` +
image-token path below or S2 will fail verification.

```bash
# 1. Dependencies + LLaVA-Med repo (installs `llava` package with helpers).
cd ${WORKSPACE_DIR}
python -m venv --system-site-packages env
source env/bin/activate
pip install -r requirements.txt
# Match CUDA build to host (CUDA 12.6 node → cu126 wheels):
# pip install torch==2.3.* torchvision --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/microsoft/LLaVA-Med.git
pip install -e LLaVA-Med
```

```python
# 2. Download weights once and load on GPU (single long-lived process).
import os, json, time, torch
from huggingface_hub import snapshot_download
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

WORKSPACE = os.environ["WORKSPACE_DIR"]
model_path = snapshot_download(
    "microsoft/llava-med-v1.5-mistral-7b",
    local_dir=os.path.join(WORKSPACE, "LLaVA-Med", "weights"),
)
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),  # "llava-med-v1.5-mistral-7b"
    load_8bit=False,
    load_4bit=False,
    device_map={"": 0},
)
model.eval()
print("model on:", model.device, "dtype:", model.dtype)
```

```python
# 3. Build the prompt via conv_templates — a bare question returns "".
from PIL import Image

def run_llava_med(question: str, image_path: str, max_new_tokens: int = 256) -> str:
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=model.dtype)

    # LLaVA-Med v1.5 mistral uses the "mistral_instruct" conversation template.
    conv = conv_templates["mistral_instruct"].copy()
    qs = f"{DEFAULT_IMAGE_TOKEN}\n{question}"   # image token MUST be in the user turn
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    # CRITICAL: decode only the newly generated tokens, not the echoed prompt.
    generated = tokenizer.batch_decode(
        output_ids[:, input_ids.shape[-1]:], skip_special_tokens=True
    )[0].strip()
    return generated
```

```python
# 4. Write the mandatory smoke_forward.json artefact.
#    Required by the scorer: success=true, wall_s>=0.3, raw_output_sample
#    must be a real non-empty decode (not "", "heuristic:", "fallback:",
#    "placeholder:").  If decode returns "", DO NOT write success=true —
#    debug prompt formatting first.
import glob

sample = sorted(glob.glob("/data/public/*/question.json"))[0]
with open(sample) as f:
    q = json.load(f)
sample_dir = os.path.dirname(sample)
image_path = os.path.join(sample_dir, q["image_paths"][0])
question_text = q.get("question") or q.get("stem") or ""

t0 = time.time()
raw = run_llava_med(question_text, image_path)
wall_s = round(time.time() - t0, 3)
print(f"smoke raw decode ({len(raw)} chars):", raw[:200])

assert raw and len(raw.strip()) >= 5, (
    f"Empty/short decode ({raw!r}). Check: (1) image token in user turn, "
    f"(2) tokenizer_image_token, (3) output_ids slicing. Do NOT write "
    f"success=true until this produces a real sentence."
)

with open(os.path.join(WORKSPACE, "smoke_forward.json"), "w") as f:
    json.dump({
        "model_name": "microsoft/llava-med-v1.5-mistral-7b",
        "device": str(model.device),
        "wall_s": wall_s,
        "raw_output_sample": raw[:200],
        "success": True,
    }, f, indent=2)
print("smoke_forward.json written — wall_s:", wall_s)
```

Common failure modes and fixes:
- **Empty decode (`raw == ""`)** — 99% of the time, `DEFAULT_IMAGE_TOKEN`
  is not in the user turn or `tokenizer_image_token` was not used. Rebuild
  the prompt through `conv_templates[...]` as above.
- **Echoed prompt in output** — you decoded the full `output_ids` instead of
  slicing `output_ids[:, input_ids.shape[-1]:]`.
- **`AttributeError: 'NoneType' object has no attribute 'to'` on images** —
  you passed `None` image tensor. Confirm `process_images(...)` returned a
  tensor before `.generate(...)`.
- **Repeated question as the answer** — the model ran fine but `conv.sep`
  stop token was not injected. Use default (LLaVA-Med's `load_pretrained_model`
  sets this up correctly); do not pass a custom `stopping_criteria` unless
  you also set `conv.sep_style`.
- **OOM at load time** — use `load_8bit=True` in `load_pretrained_model`,
  or split across two GPUs with `device_map="auto"`.
