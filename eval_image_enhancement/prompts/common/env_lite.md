## Environment
You start with a base environment containing:
Python 3, PyTorch 2.6 + CUDA, NumPy, SciPy, scikit-image, nibabel, pandas, \
huggingface_hub.
Use this base environment for research in S1. In S2, create a virtual \
environment (python -m venv --system-site-packages) under {output_dir}/env/ \
and install dependencies from the provided requirements file:
  {requirements_txt_path}
You may install additional packages if needed. Do NOT install packages \
into the base environment.
