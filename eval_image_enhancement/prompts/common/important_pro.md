## Important
- This is an INFERENCE-ONLY benchmark. Find pre-trained models and load their \
weights, OR use classical analytical methods — do NOT train or fine-tune any model.
- You MUST NOT call `.backward()`, `loss.backward()`, `optimizer.step()`, \
`optimizer.zero_grad()`, `model.train()`, or any training-mode method. Any \
weight update is a protocol violation and will score 0 on clinical.
- If your chosen architecture has no public pre-trained checkpoint, fall back \
to a classical analytical method (BM3D, bilateral filter, non-local means) — \
do NOT train from scratch on the data you see here.
- You MUST NOT access any file under `private/` or read any `reference.npy` — \
all references are held out for scoring.
- Enhanced outputs MUST be float32 NumPy arrays with the same spatial shape as \
the input, no NaN/Inf, and values within the expected clinical intensity \
range of the modality.
- Process ALL patients found in the data directory — missing ANY patient means \
automatic failure (Rating F, zero clinical credit).
- You are competing against other agents. The winning strategy is NOT speed — \
it is finding the best enhancement approach. An agent that spends 5 minutes \
on research and picks a mediocre method will lose to one that spends 15 \
minutes and finds the right one.
- Over-smoothing will hurt LPIPS and SSIM. The best enhancer preserves \
clinically meaningful structure (edges, small lesions) while suppressing noise.
- Print progress so the log captures what's happening.
