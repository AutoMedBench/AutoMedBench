## Lite S2 Skill

Set up the environment under `{output_dir}/env/`, then follow the fixed YOLOv8s route:

1. Install the dependencies needed for the article workflow, especially
   `ultralytics`
2. Download the **Baidu Netdisk archive referenced in the article**
3. Unpack it under `{output_dir}/model/`
4. Locate the usable YOLOv8s inference weights inside the unpacked project
   (for example a `best.pt`-style checkpoint)
5. Confirm a single-image YOLO forward pass works before moving on

Do not replace this with a different checkpoint family or a different repo.
