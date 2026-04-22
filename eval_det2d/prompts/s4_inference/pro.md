S4  INFERENCE   Run inference on ALL patients.
  1. Run the pipeline on every patient in the dataset. Save `prediction.json` \
under agents_outputs/<patient_id>/ for each patient.
  2. Make full use of available computational resources for maximum throughput.
  3. If the pipeline supports batched inference, use it.
  4. Print progress so the log captures which patient is being processed.
  5. After all patients are done, confirm that every patient has output JSON.
  Do NOT skip JSON validation before S5.
