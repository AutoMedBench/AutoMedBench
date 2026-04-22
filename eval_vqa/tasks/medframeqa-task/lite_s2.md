Use the provided `requirements.txt` as the starting dependency list. Confirm:

- PyTorch, torchvision, and torchaudio come from the official PyTorch wheel
  index with a CUDA build that matches the host driver. On the current
  `560.35.03` / CUDA `12.6` node, use the `cu126` wheels instead of a newer
  `cu130` build.
- model weights and processor assets download locally
- the environment can import the required transformer classes
- one GPU forward pass succeeds on at least one staged sample
