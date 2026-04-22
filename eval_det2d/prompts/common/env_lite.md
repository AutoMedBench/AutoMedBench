## Environment
You start with a base environment containing:
Python 3, PyTorch 2.6 + CUDA, torchvision, numpy, scipy, scikit-image, \
pandas, Pillow, pydicom, requests, PyYAML.
Use this base environment for research in S1. In S2, create a virtual \
environment (python -m venv) under {output_dir}/env/ and install \
dependencies from the provided requirements file:
  {requirements_txt_path}
For this lite task, the intended route is the article-provided YOLOv8s setup, \
so your environment should be compatible with `ultralytics` inference and the \
downloaded archive contents. Do NOT switch to a different detector stack unless \
the fixed article route is genuinely unusable.
