S1  RESEARCH    Find and understand the enhancement method you will use.
  Your target method is **{model_architecture}**. {model_description}
  1. If the method requires pre-trained weights, download them from the \
specified source. If the method is classical analytical (BM3D, bilateral \
filter, non-local means, etc.), install the Python package that implements it.
  2. CRITICAL: Verify that the method you pick is a valid inference-only \
enhancement approach. Training, fine-tuning, or any weight update is \
DISALLOWED. If a checkpoint is unavailable, do NOT train — fall back to a \
classical analytical method.
  3. Check the expected input format (intensity range, normalization, 2D vs \
channels, dtype).
  4. Save your findings to {output_dir}/plan/plan.md — include: method name, \
weight source (or "classical, no weights"), input preprocessing, inference \
pipeline, and how you will invert any normalization before saving the output.
