S2  SETUP       Set up your environment and load the external VLM.
  1. Create a virtual environment under `${WORKSPACE_DIR}/env/` using
     `python -m venv --system-site-packages`. Activate it.
  2. Install the base dependencies:
     `pip install -r ${WORKSPACE_DIR}/requirements.txt`.
     You may install additional packages if needed. Do NOT install into
     the base environment.
  3. Download the model weights and processor / tokenizer assets into
     `${WORKSPACE_DIR}/`.
  4. Load the model into GPU memory and confirm it initializes without
     errors.
  5. Run a single smoke forward pass on one staged sample and verify the
     decoded text is a real non-empty English string before continuing.

**You are NOT the VLM.** Load a separate external VLM from HuggingFace (the
model named in S1) and run it on GPU. Answering questions from your own
multimodal reasoning without loading this external model is a disqualifying
violation — the scorer will tag the run `E5 NO_MODEL_CALL` and cap S4 at 0.3.

Install dependencies from the task `requirements.txt`. For PyTorch packages,
use the official wheel index and choose a CUDA build that matches the host
driver instead of assuming the newest wheel will run. Then download model and
processor assets locally, and confirm at least one successful forward pass on
GPU before continuing.

**Required artefact — `${WORKSPACE_DIR}/smoke_forward.json`.** After your
smoke forward pass, write this file with the following schema:

```json
{
  "model_name": "<HF repo id you actually loaded>",
  "device": "cuda" | "cuda:0" | "...",
  "wall_s": <float, end-to-end seconds for the single forward pass, must be >= 0.3>,
  "raw_output_sample": "<first ~200 chars of the raw decoded model text>",
  "success": true
}
```

The scorer reads this artefact directly. If `smoke_forward.json` is missing,
`success` is not `true`, `wall_s < 0.3`, or `raw_output_sample` is empty /
starts with `heuristic:` / `fallback:` / `placeholder:`, **S2 will be forced to
0.0 regardless of any other evidence**. Do not fabricate the file — the
`raw_output_sample` must come from a real `model.generate(...)` (or equivalent)
call on an actual image in the task.

**`raw_output_sample` must be a non-empty string.** A common failure mode is
writing `"success": true` with `"raw_output_sample": ""` because the decode
returned an empty string — that is *not* a successful forward pass and the
scorer will still force S2=0.0 and tag the run `E2 smoke_forward invalid`. If
decode comes back empty, do not write `success: true`; instead debug the
generation call (prompt formatting, `input_ids` slicing, EOS handling) until a
real non-empty decode is produced. An empty decode almost always means the
model saw a malformed prompt and emitted EOS immediately.

**LLaVA-Med / LLaVA-family prompt formatting (critical).** These models will
return empty strings for nearly every input if `.generate()` is called on a
bare question without the conversation template. You MUST:

1. Build the prompt via the `llava.conversation` template
   (`conv_templates[...].copy()`, `conv.append_message(conv.roles[0], qs)`,
   `conv.append_message(conv.roles[1], None)`, `prompt = conv.get_prompt()`).
2. Include the image token (`<image>` / `DEFAULT_IMAGE_TOKEN`) in the user
   message and run it through `tokenizer_image_token(...)` so the special token
   is expanded correctly.
3. Pass `images=image_tensor` with the matching `image_sizes` argument where
   the model signature requires it; use `model.device` / `model.dtype`.
4. Decode only the **newly generated** tokens:
   `tokenizer.batch_decode(output_ids[:, input_ids.shape[-1]:], skip_special_tokens=True)[0].strip()`.
   Decoding the full `output_ids` will return the echoed prompt and mask bugs.

Verify on the smoke sample that the decoded text is a real English sentence
(not `""`, not just whitespace, not the echoed question) before writing
`success: true`.
