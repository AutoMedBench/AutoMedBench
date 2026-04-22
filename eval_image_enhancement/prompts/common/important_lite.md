## Important
- This is an INFERENCE-ONLY benchmark. Load pre-trained weights OR use classical \
analytical methods — do NOT train or fine-tune any model.
- You MUST NOT call `.backward()`, `loss.backward()`, `optimizer.step()`, \
`optimizer.zero_grad()`, `model.train()`, or any training-mode method. Any \
weight update is a protocol violation and will score 0 on clinical.
- You MUST NOT access any file under `private/` or read any `reference.npy` — \
all references are held out for scoring. You only see noisy inputs under the \
data directory.
- Enhanced outputs MUST be float32 NumPy arrays with the same spatial shape as \
the input, no NaN/Inf, and values within the expected clinical intensity \
range of the modality.
- Process ALL patients found in the data directory — missing ANY patient means \
automatic failure (Rating F, zero clinical credit).
- Print progress so the log captures what's happening.
