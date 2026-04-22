S4  INFERENCE   Run inference on ALL patients.
  1. Run the enhancement pipeline on every patient in {data_dir}/. Save \
`enhanced.npy` under `agents_outputs/<patient_id>/` for each patient.
  2. Make full use of available computational resources for maximum throughput.
  3. If the pipeline supports batched inference, use it.
  4. Print progress so the log captures which patient is being processed.
  5. After all patients are done, confirm every patient has an \
`enhanced.npy` under `agents_outputs/<patient_id>/`.
  Do NOT apply heuristic post-processing to the outputs. Use the raw method \
outputs directly.
