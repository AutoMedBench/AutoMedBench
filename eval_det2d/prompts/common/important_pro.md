## Important
- This is an INFERENCE-ONLY benchmark. Find pre-trained models and load their \
weights — do NOT train or fine-tune any model.
- Outputs MUST be valid `prediction.json` files with in-bounds coordinates.
- Process ALL patients found in the data directory — missing ANY patient means automatic failure (Rating F, zero task credit).
- If a model does not output bounding boxes directly, consider alternative \
approaches (e.g. different detector, lightweight conversion logic, or a small ensemble).
- You are competing against other agents. The winning strategy is NOT speed — \
it is finding the best model. An agent that spends 5 minutes on research \
and picks a mediocre model will lose to one that spends 15 minutes and \
finds the right model.
- Classification-only models will score poorly. Box quality is the decisive metric.
- Print progress so the log captures what's happening.
