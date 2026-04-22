S4  INFERENCE   Run inference on ALL patients.
  1. Run the enhancement pipeline on every patient in {data_dir}/. Save \
`enhanced.npy` under `agents_outputs/<patient_id>/` for each patient.
  2. For DNN methods, you MUST use GPU (CUDA) for inference. Keep the model \
in `eval` mode. Never call `.train()`. Never compute gradients.
  3. Maximize throughput — if the pipeline supports batched inference, use \
it. For classical methods, parallelize across patients if helpful.
  4. Print progress so the log captures which patient is being processed.
  5. After all patients are done, confirm that every patient has an \
`enhanced.npy` under `agents_outputs/<patient_id>/`.
  Do NOT apply heuristic post-processing (histogram matching, tone-mapping, \
etc.) to the outputs. Use the raw method outputs directly.
