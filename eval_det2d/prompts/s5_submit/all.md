S5  SUBMIT      Verify outputs and submit results.
  1. Confirm that every patient has `prediction.json` \
under agents_outputs/<patient_id>/.
  2. Verify no files are missing or corrupted.
  3. Spot-check at least one JSON file to ensure the coordinates remain within image bounds.
  4. Call `submit_results` when everything is saved and verified.
