S2  SETUP       Set up your environment and prepare the method for inference.
  1. Create a virtual environment under {output_dir}/env/ using \
`python -m venv --system-site-packages`. Activate it.
  2. Install the base dependencies: `pip install -r {requirements_txt_path}`
     You may install additional packages if needed.
  3. If your method requires pretrained weights, download them now. If it is \
a classical analytical method, install the corresponding package \
(e.g., `pip install bm3d`).
  4. Verify the method can run on a dummy 2D array of the expected shape. \
For DNN methods, load the model onto GPU and confirm a forward pass works. \
For classical methods, run the function on a synthetic input and confirm \
it returns the expected shape and dtype.
  5. REMINDER: do NOT set `model.train()`, do NOT create an optimizer, do \
NOT call `.backward()` anywhere. This is inference-only.

  Skill — How to set up the environment (examples only — use any approach that works):
  ```bash
  python -m venv --system-site-packages {output_dir}/env
  source {output_dir}/env/bin/activate
  pip install -r {requirements_txt_path}
  ```

  ```python
  # For classical methods:
  import bm3d, numpy as np
  dummy = np.random.randn(512, 512).astype(np.float32)
  denoised = bm3d.bm3d(dummy, sigma_psd=50.0)
  print(denoised.shape, denoised.dtype)

  # For DNN methods:
  from huggingface_hub import snapshot_download
  model_dir = snapshot_download("REPO_ID", local_dir="{output_dir}/model")
  import torch
  model = ...  # model-specific loading
  model.eval()  # inference mode, never .train()
  model.cuda()
  with torch.no_grad():
      out = model(torch.randn(1, 1, 512, 512).cuda())
  print(out.shape)
  ```
