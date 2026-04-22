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
print(f"Model loaded on {device}")
```
