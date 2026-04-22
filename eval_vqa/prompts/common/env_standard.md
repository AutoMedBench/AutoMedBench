## Environment
You start with a base Python environment containing PyTorch + CUDA and the
common medical imaging / HF stack. Use the base environment directly — if you
need extra packages for a chosen candidate model, `pip install` them into the
base environment. Internet access is available for weight downloads and pip.

Cache and download rules:
- put Hugging Face, pip, torch, and tmp caches under `{output_dir}/.cache/`
- set `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `PIP_CACHE_DIR` to subpaths of that cache dir before any download
- do not write caches to `/root`, `/tmp`, or home directories
