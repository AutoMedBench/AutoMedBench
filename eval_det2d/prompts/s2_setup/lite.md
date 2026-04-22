S2  SETUP       Set up your environment and load the model.
  1. Create a virtual environment under {output_dir}/env/ using \
`python -m venv --system-site-packages`. Activate it.
  2. Install the base dependencies: `pip install -r {requirements_txt_path}`
     This lite task is tied to the article-specified YOLOv8s route, so make \
     sure `ultralytics` is available.
  3. Download the Baidu Netdisk archive referenced by the fixed article route, \
     unpack it under {output_dir}/model/, and identify the usable inference \
     checkpoint.
  4. Load the YOLOv8s detector into GPU memory and confirm it initializes without errors.
  5. Verify compatibility with the image data (check one image's size, dtype, \
normalization, and channel format against model expectations).

  Skill — How to set up the environment (examples only — use any approach that works):
  ```bash
  # Create venv with access to system packages
  python -m venv --system-site-packages {output_dir}/env
  source {output_dir}/env/bin/activate

  # Install base dependencies
  pip install -r {requirements_txt_path}
  ```

  ```python
  from ultralytics import YOLO

  # Example structure only — locate the real checkpoint from the downloaded archive.
  model = YOLO("{output_dir}/model/path/to/best.pt")
  results = model.predict(source="{data_dir}/SAMPLE_PATIENT/image.png", device=0)
  print(results[0].boxes.xyxy.shape if results else "no results")
  ```
