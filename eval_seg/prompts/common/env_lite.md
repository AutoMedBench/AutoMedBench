## Environment
You start with a base environment containing:
Python 3, PyTorch 2.6 + CUDA, MONAI 1.5.0 (monai.bundle for model zoo), \
nnU-Net v2.6.0, nibabel, numpy, scipy, scikit-image, pandas.
Use this base environment for research in S1. In S2, create a virtual \
environment (python -m venv) under {output_dir}/env/ and install \
dependencies from the provided requirements file:
  {requirements_txt_path}
You may install additional packages if needed. Do NOT install packages \
into the base environment.
