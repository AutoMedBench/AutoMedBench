S4  INFERENCE   Run inference on ALL patients.
  1. Run the pipeline on every patient in the dataset. Save organ.nii.gz \
and lesion.nii.gz under agents_outputs/<patient_id>/ for each patient.
  2. Make full use of available computational resources for maximum throughput.
  3. If the pipeline supports batched inference, use it.
  4. Print progress so the log captures which patient is being processed.
  5. After all patients are done, confirm that every patient has both output files.
  Do NOT apply post-processing to the masks. Use the raw model outputs directly.
