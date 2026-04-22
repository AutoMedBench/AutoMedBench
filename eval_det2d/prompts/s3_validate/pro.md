S3  VALIDATE    Run inference on ONE image as a sanity check.
  1. Pick one patient and run the full inference pipeline end-to-end.
  2. Make full use of available computational resources (check nvidia-smi).
  3. Verify the output:
     - `prediction.json` parses correctly
     - Every box has `class`, `score`, `x1`, `y1`, `x2`, `y2`
     - Coordinates stay within image bounds
     - The model is not returning empty predictions on obviously positive examples
  4. If the output looks wrong or boxes are invalid, debug and fix \
the pipeline before proceeding. Do NOT continue to S4 with a model that \
produces malformed or empty predictions.
  5. If you have been stuck debugging for more than 600 seconds without a \
working pipeline, abandon the current approach and go back to S1 to choose \
a different model or method.
