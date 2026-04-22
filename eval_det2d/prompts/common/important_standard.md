## Important
- This is an INFERENCE-ONLY benchmark. Find pre-trained models and load their \
weights — do NOT train or fine-tune any model.
- Outputs MUST be valid `prediction.json` files with in-bounds coordinates.
- Process ALL patients found in the data directory — missing ANY patient means automatic failure (Rating F, zero task credit).
- If a model does not output bounding boxes directly, consider alternative \
approaches (e.g. different detector, lightweight conversion logic, or a small ensemble).
- Print progress so the log captures what's happening.
