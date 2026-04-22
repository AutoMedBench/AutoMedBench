S3  VALIDATE    Run inference on ONE patient as a sanity check.
  1. Pick one patient from {data_dir}/ and run the full enhancement pipeline \
end-to-end on that patient's `{input_filename}`.
  2. For DNN methods, you MUST use GPU (CUDA) for inference. Verify with \
`torch.cuda.is_available()`. Keep the model in `eval` mode — if you see \
`.train()` anywhere in your code, REMOVE IT.
  3. Verify the output:
     - Shape matches the input exactly (2D, same H×W).
     - Dtype is float32.
     - No NaN / Inf values.
     - Values fall within the expected clinical intensity range of the modality \
(for CT in HU: roughly [-1024, +3000]; for normalized MRI: roughly [0, 1]).
     - CRITICAL — compare the output to the INPUT (you do NOT have the reference). \
Print: input mean, input std, output mean, output std, mean abs(input - output). \
A reasonable denoiser should change the image but not replace it; ratios that are \
wildly different (e.g., output std < 10% of input std) suggest over-smoothing, and \
outputs identical to input suggest the method did nothing.
  4. If any check fails, debug and fix the pipeline before proceeding.
  5. If you have been stuck debugging for more than 600 seconds without a \
working pipeline, abandon the current approach and go back to S1 to choose a \
different method.
