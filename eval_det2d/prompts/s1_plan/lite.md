S1  RESEARCH    Find and understand the model you will use.
  Your target model is **{model_architecture}**. {model_description}
  1. Download the pre-trained weights from the specified source.
  2. CRITICAL: Verify the checkpoint is a DETECTOR that outputs bounding boxes, \
not just image-level labels. Inspect the repo, config, or inference examples.
  3. Check its expected input format (image size, normalization, channel handling).
  4. Save your findings to {output_dir}/plan/plan.md — include: model name, \
checkpoint source, output format, and inference pipeline.
