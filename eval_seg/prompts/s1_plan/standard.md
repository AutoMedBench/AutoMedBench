S1  PLAN        Research and choose the best model from the following families.
  Explore these model families for {organ} organ and TUMOR segmentation:
{model_range}
  1. For each family, search for available pre-trained models. Find specific \
checkpoints that can be downloaded.
  2. For each candidate found, verify via code: (a) weights can be downloaded \
in this environment, (b) its label map covers {organ} organ AND {organ} \
TUMOR structures. CRITICAL: general-purpose organ segmentation models \
(e.g., TotalSegmentator) typically do NOT have tumor labels and will score \
near zero on lesion Dice. You need a model trained specifically on \
tumor-annotated data.
  3. Compare candidates on tumor label coverage, reported accuracy, setup \
complexity, and feasibility in this environment.
  4. Select the best option. Justify your choice — explain why it beats the \
alternatives for this specific task. Prioritize tumor detection capability.
  5. Save your plan to {output_dir}/plan/plan.md — include: candidates \
evaluated, comparison table, final decision, and the pipeline you will \
follow (which model for organ, which for tumor, preprocessing steps).
  6. Generate a flowchart visualization of your plan and save it as \
{output_dir}/plan/plan.png (use matplotlib or any plotting library).

  Skill — How to search and compare models (examples only — use any approach that works):
  ```python
  # Search HuggingFace for models
  import requests
  resp = requests.get("https://huggingface.co/api/models",
                      params={{"search": "{organ} tumor segmentation", "limit": 10}})
  for model in resp.json():
      print(model["modelId"], model.get("tags", []))

  # Download model weights from a URL
  import urllib.request
  urllib.request.urlretrieve(
      "https://example.com/model/weights.zip",
      "{output_dir}/model/weights.zip")

  # Check MONAI Model Zoo bundles
  from monai.bundle import download
  # Try: monai.bundle.download("bundle_name", bundle_dir="{output_dir}/model")
  # Inspect: configs/inference.json, docs/labels.json for label maps
  ```

  ```markdown
  # Comparison table template for plan.md:
  | Model | Organ Labels | TUMOR Labels | Download Size | Ease of Setup | Notes |
  |-------|-------------|--------------|---------------|---------------|-------|
  | ...   | ...         | ...          | ...           | ...           | ...   |
  ```
