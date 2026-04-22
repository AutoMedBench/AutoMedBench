S5  SUBMIT      Verify outputs and submit results.
  1. Confirm that every patient has both organ.nii.gz and lesion.nii.gz \
under agents_outputs/<patient_id>/.
  2. Verify no files are missing or corrupted.
  3. (Optional) Write a decision CSV to {output_dir}/agents_outputs/agents_decision.csv \
with columns: patient_id, organ, lesion_present (0 or 1). One row per patient. \
Set lesion_present=1 if the lesion mask has non-trivial foreground voxels, 0 otherwise. \
This CSV is not scored but is useful for analysis.
  4. Call `submit_results` when everything is saved and verified.
