Skill — How to search and compare models (examples only — use any approach that works):
```python
# Search HuggingFace for models
import requests
resp = requests.get("https://huggingface.co/api/models",
                    params={"search": "{organ} tumor segmentation", "limit": 10})
for model in resp.json():
    print(model["modelId"], model.get("tags", []))

# Download model weights from a URL
import urllib.request
urllib.request.urlretrieve(
    "https://example.com/model/weights.zip",
    "{output_dir}/model/weights.zip")

# Check MONAI Model Zoo bundles
from monai.bundle import download
# Try: monai.bundle.download("bundle_name", bundle_dir="{output_dir}/model")
# Inspect: configs/inference.json, docs/labels.json for label maps
```

```markdown
# Comparison table template for plan.md:
| Model | Organ Labels | TUMOR Labels | Download Size | Ease of Setup | Notes |
|-------|-------------|--------------|---------------|---------------|-------|
| ...   | ...         | ...          | ...           | ...           | ...   |
```
