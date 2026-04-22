S5  SUBMIT      Verify outputs and submit results.
  1. Confirm that every patient has `enhanced.npy` under \
`agents_outputs/<patient_id>/`.
  2. Verify no files are missing or corrupted (np.load() each, check shape \
and no NaN/Inf).
  3. (Optional) Write a decision CSV to `{output_dir}/agents_outputs/agents_decision.csv` \
with columns: patient_id, method, notes. One row per patient. This CSV is \
not scored but is useful for analysis.
  4. Call `submit_results` when everything is saved and verified.
