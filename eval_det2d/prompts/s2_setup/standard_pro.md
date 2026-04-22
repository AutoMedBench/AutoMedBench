S2  SETUP       Set up your environment and download the chosen model(s).
  1. Create a virtual environment for this run under {output_dir}/env/ \
using `python -m venv`. Activate it and install any additional packages \
your chosen model requires. Use this venv for all subsequent steps.
  2. Download model weights, configs, and any required dependencies.
  3. Load the model into GPU memory and confirm it initializes without errors.
  4. Check the model's expected input format (image size, color mode, normalization) \
and verify compatibility with the image data.
