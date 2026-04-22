S3  VALIDATE    Run inference on ONE patient as a sanity check.
  1. Pick one patient and run the full enhancement pipeline end-to-end.
  2. Make full use of available computational resources (check nvidia-smi).
  3. Verify the output:
     - Shape matches the input exactly (2D, same H×W).
     - Dtype is float32.
     - No NaN / Inf values.
     - Values fall within the expected clinical intensity range of the modality.
     - Compare the output to the INPUT (you do NOT have the reference). \
Print input stats, output stats, and mean abs(input - output). Interpret \
whether the method is making a meaningful change and not over-smoothing.
  4. If any check fails, debug and fix the pipeline before proceeding.
  5. If stuck for more than 600 seconds without a working pipeline, go back \
to S1 and choose a different method.
