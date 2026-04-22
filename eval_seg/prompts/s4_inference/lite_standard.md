S4  INFERENCE   Run inference on ALL patients.
  1. Run the pipeline on every patient in the dataset. Save organ.nii.gz \
and lesion.nii.gz under agents_outputs/<patient_id>/ for each patient.
  2. You MUST use GPU (CUDA) for inference — the same as S3. Ensure model \
and data are on GPU. Never use map_location='cpu' when a GPU is available.
  3. Maximize throughput — if the pipeline supports batched inference, use it.
  4. Print progress so the log captures which patient is being processed.
  5. After all patients are done, confirm that every patient has both output files.
  Do NOT apply post-processing to the masks. Use the raw model outputs directly.
