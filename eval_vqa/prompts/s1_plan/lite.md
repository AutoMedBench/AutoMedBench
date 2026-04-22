S1  RESEARCH    Find and understand the fixed model you will use.

  Your target model is the **fixed lite-tier VLM** named in the
  `Tier/model guidance` block below (e.g. `microsoft/llava-med-v1.5-mistral-7b`
  for LLaVA-Med tasks). The per-task `lite_s1.md` skill ships the exact
  download + inspection code.

  1. Download the pre-trained weights from HuggingFace into
     `${WORKSPACE_DIR}/`. Use `snapshot_download(...)` or the helper named in
     the skill. Do not re-download if the weights already exist.
  2. Verify the model's input contract: which conversation template to use
     (e.g. `conv_templates["mistral_instruct"]` for LLaVA-Med v1.5), where the
     image token belongs in the user turn, and how to slice
     `output_ids[:, input_ids.shape[-1]:]` so the decode does not echo the
     prompt. Mismatched prompt format is the top reason `.generate(...)`
     returns an empty string and S2 fails verification.
  3. Inspect the staged data layout at
     `{data_dir}/<question_id>/question.json` — record which fields are
     present (`question`, `image_paths`, `options`, `valid_labels`,
     `question_type`), and note whether samples are single-image or
     multi-frame.
  4. Save your findings to `${WORKSPACE_DIR}/plan/plan.md` — include: model
     name + HF repo id, conversation template, image-token placement,
     answer-schema plan (`predicted_label` for MCQ / `predicted_answer` for
     open-ended), and which `medbench_vqa` helpers
     (`inspect_image`, `public_medical_search`, `submit_answer`) you will
     call in which stage.
