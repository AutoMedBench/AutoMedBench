## Important
- This is an INFERENCE-ONLY benchmark. Load pre-trained weights — do NOT \
train or fine-tune any model.
- Masks MUST be binary (0 and 1 only) and match the CT spatial dimensions exactly.
- Process ALL patients found in the data directory — missing ANY patient means automatic failure (Rating F, zero Dice credit).
- Print progress so the log captures what's happening.
