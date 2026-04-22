## Important
- This is an INFERENCE-ONLY benchmark. Load pre-trained weights — do NOT \
train or fine-tune any model.
- Outputs MUST be valid `prediction.json` files with a `boxes` list.
- Every predicted box must include `class`, `score`, `x1`, `y1`, `x2`, `y2`.
- Coordinates must remain within the image bounds.
- Process ALL patients found in the data directory — missing ANY patient means automatic failure (Rating F, zero task credit).
- Print progress so the log captures what's happening.
