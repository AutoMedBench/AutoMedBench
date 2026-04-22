## Environment
You start with a base Python environment containing PyTorch + CUDA and the
common medical imaging / HF stack. Use the base environment for S1 research.
In S2, create a virtual environment under `{output_dir}/env/` (`python -m venv`)
and install the task requirements file that was copied to `{output_dir}/requirements.txt`.
You may install additional packages as needed. Do NOT install packages into
the base environment.

Cache and download rules:
- put Hugging Face, pip, torch, and tmp caches under `{output_dir}/.cache/`
- set `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `PIP_CACHE_DIR` to subpaths of that cache dir before any download
- do not write caches to `/root`, `/tmp`, or home directories
