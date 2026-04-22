S1  PLAN        Research and choose the best model from the following families.
  Explore these model families for chest X-ray abnormality detection:
{model_range}
  1. For each family, search for available pre-trained models. Find specific \
checkpoints that can be downloaded.
  2. For each candidate found, verify via code: (a) weights can be downloaded \
in this environment, (b) it outputs 2D bounding boxes, and (c) it is usable \
for chest X-ray localization. Image-level classifiers are not sufficient.
  3. Compare candidates on localization quality, reported accuracy, setup \
complexity, and feasibility in this environment.
  4. Select the best option. Justify your choice — explain why it beats the \
alternatives for this specific task. Prioritize usable box localization.
  5. Save your plan to {output_dir}/plan/plan.md — include: candidates \
evaluated, comparison table, final decision, and the pipeline you will \
follow (model, preprocessing, postprocessing, JSON export steps).
  6. Generate a flowchart visualization of your plan and save it as \
{output_dir}/plan/plan.png (use matplotlib or any plotting library).

  Skill — How to search and compare models (examples only — use any approach that works):
  ```python
  # Search HuggingFace for models
  import requests
  resp = requests.get("https://huggingface.co/api/models",
                      params={{"search": "vindr cxr detection", "limit": 10}})
  for model in resp.json():
      print(model["modelId"], model.get("tags", []))

  # Download model weights from a URL
  import urllib.request
  urllib.request.urlretrieve(
      "https://example.com/model/weights.zip",
      "{output_dir}/model/weights.zip")

  # Check a repo or config to confirm it outputs bounding boxes
  # and not only image-level labels
  ```

  ```markdown
  # Comparison table template for plan.md:
  | Model | Output Type | Checkpoint | Download Size | Ease of Setup | Notes |
  |-------|-------------|------------|---------------|---------------|-------|
  | ...   | ...         | ...          | ...           | ...           | ...   |
  ```
