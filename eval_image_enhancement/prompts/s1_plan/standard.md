S1  PLAN        Research and choose the best enhancement method from the following candidates.
  Candidate methods for {modality} {task_type}:
{model_range}
  1. For each candidate, determine whether it needs pre-trained weights. If it \
does, verify a public checkpoint is actually downloadable. If it does not \
(classical method), install the corresponding package.
  2. CRITICAL: Training, fine-tuning, or any weight update is DISALLOWED. You \
must pick an inference-only path. If a candidate needs training data to work, \
either (a) find a pretrained checkpoint for it, or (b) drop it from consideration.
  3. Compare candidates on: public checkpoint availability, reported denoising/SR \
quality (PSNR/SSIM/LPIPS from the literature), inference-time complexity, \
setup difficulty.
  4. Select the best option. Justify your choice — explain why it beats the \
alternatives for this specific task.
  5. Save your plan to {output_dir}/plan/plan.md — include: candidates \
evaluated, comparison table, final decision, and the pipeline you will \
follow (preprocessing, inference, postprocessing).
  6. Generate a flowchart visualization of your pipeline and save it as \
{output_dir}/plan/plan.png (use matplotlib or any plotting library).

  Skill — How to search and compare methods (examples only — use any approach that works):
  ```python
  # Search HuggingFace for restoration models
  import requests
  resp = requests.get("https://huggingface.co/api/models",
                      params={{"search": "denoising medical", "limit": 10}})
  for m in resp.json():
      print(m["modelId"], m.get("tags", []))

  # Download a weights repo from HuggingFace
  from huggingface_hub import snapshot_download
  model_dir = snapshot_download("REPO_ID", local_dir="{output_dir}/model")
  ```

  ```markdown
  # Comparison table template for plan.md:
  | Method | Type (DNN/classical) | Public weights | Reported PSNR | Setup effort | Notes |
  |--------|----------------------|----------------|---------------|--------------|-------|
  | ...    | ...                  | ...            | ...           | ...          | ...   |
  ```
