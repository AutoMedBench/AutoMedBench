S1  PLAN        Research and choose the best model from the candidate families.

  The per-task `standard_s1.md` skill lists the candidate models for this
  task (typically an open medical VLM, a general Qwen2.5-VL / Gemma baseline,
  and one gated fallback).

  1. For each candidate family, check: (a) weights downloadable in this
     environment (no auth wall), (b) conversation template / image-token
     contract is documented, (c) single-GPU inference is feasible at this
     task's context budget.
  2. Verify via code by loading at least one candidate and running a single
     forward pass on one staged sample. Candidates that silently return `""`
     from `.generate(...)` under their default pipeline are unusable — note
     them and move on.
  3. Compare candidates on: open vs. gated accessibility, medical-domain
     pretraining, prompt-format stability, reported MCQ / open-ended accuracy
     on similar datasets. Prioritize open medical VLMs over general VLMs;
     treat Gemma variants as last-resort (gated).
  4. Select the best option. Justify your choice — explain why it beats the
     alternatives for this specific task (e.g. "MedVLThinker ships a
     documented Qwen2.5-VL chat template and passes a smoke forward on our
     sample in 2.1s; LLaVA-Med v1.6 gated, dropped").
  5. Save your plan to `${WORKSPACE_DIR}/plan/plan.md` — include: candidates
     evaluated, a short comparison table, the final decision, and the
     pipeline you will follow (prompt template, image-token handling, MCQ
     parsing or open-ended postprocess plan).
  6. Inspect the staged data layout at
     `{data_dir}/<question_id>/question.json` and record the field names
     you will depend on in S3/S4.
  7. Generate a flowchart visualization of your plan and save it as
     `${WORKSPACE_DIR}/plan/plan.png` (use matplotlib or any plotting
     library). The flowchart should show: candidate shortlist → chosen
     VLM → prompt / image-token pipeline → postprocess → submission.
