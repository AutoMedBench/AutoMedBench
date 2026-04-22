S2  SETUP       Set up your environment and prepare the chosen method.
  1. Create a virtual environment for this run under {output_dir}/env/ using \
`python -m venv`. Activate it. Install any additional packages your chosen \
method requires. Use this venv for all subsequent steps.
  2. If your method uses a DNN, download the pretrained weights now. If it \
is a classical method, install the corresponding package.
  3. Load the method and confirm it initializes without errors. For DNN, \
put it on GPU in `eval` mode — NEVER call `.train()`.
  4. Check the method's expected input format (intensity range, normalization, \
2D vs channels, dtype) and verify compatibility with the data in this dataset.
  5. REMINDER: INFERENCE-ONLY. Do NOT create an optimizer. Do NOT call \
`.backward()`. Any weight update is a protocol violation.
