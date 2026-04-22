## Important
- This is an INFERENCE-ONLY benchmark. Find pre-trained models and load their \
weights — do NOT train or fine-tune any model.
- Masks MUST be binary (0 and 1 only) and match the CT spatial dimensions exactly.
- Process ALL patients found in the data directory — missing ANY patient means automatic failure (Rating F, zero Dice credit).
- If a model does not have a {organ} lesion/tumor class, consider alternative \
approaches (e.g. different model, combining models, or using available labels as proxy).
- Print progress so the log captures what's happening.
