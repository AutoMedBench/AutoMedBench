S1  RESEARCH    Find and understand the model you will use.
  Your target model is **{model_architecture}**. {model_description}
  1. Download the pre-trained weights from the specified source.
  2. CRITICAL: Verify the checkpoint has TUMOR labels (not just organ or \
cyst labels). Inspect the label map. You need a model trained specifically \
on tumor-annotated data.
  3. Check its expected input format (spacing, orientation, intensity range).
  4. Save your findings to {output_dir}/plan/plan.md — include: model name, \
checkpoint source, label map, and inference pipeline.
