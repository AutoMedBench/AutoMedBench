S2  SETUP       Set up your environment and load the model.
  1. Create a virtual environment under {output_dir}/env/ using \
`python -m venv --system-site-packages`. Activate it.
  2. Install the base dependencies: `pip install -r {requirements_txt_path}`
     You may install additional packages if needed.
  3. Download model weights and any required configs.
  4. Load the model into GPU memory and confirm it initializes without errors.
  5. Verify compatibility with the scan data (check one scan's shape, spacing, \
and intensity range against model expectations).

  Skill — How to set up the environment (examples only — use any approach that works):
  ```bash
  # Create venv with access to system packages (PyTorch, MONAI, etc.)
  python -m venv --system-site-packages {output_dir}/env
  source {output_dir}/env/bin/activate

  # Install base dependencies
  pip install -r {requirements_txt_path}

  # Install additional packages if needed
  pip install <package_name>
  ```

  ```python
  # Download model from HuggingFace
  from huggingface_hub import snapshot_download
  model_dir = snapshot_download("REPO_ID", local_dir="{output_dir}/model")

  # Load and verify on GPU
  import torch
  device = torch.device("cuda")
  model = ...  # model-specific loading
  model.to(device)
  model.eval()
  print(f"Model loaded on {{device}}")
  ```
