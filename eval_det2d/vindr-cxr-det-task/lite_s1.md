## Lite S1 Skill

You are solving **2D object detection on chest X-rays**.

Your model choice is already fixed:

- Use the **YOLOv8s** VinDr-CXR route from the CSDN article
  `https://blog.csdn.net/weixin_39381937/article/details/139466170`
- Use the **Baidu Netdisk link provided in that article**
- Do **not** switch to any other model family

This is still a detection task, so the model must output **bounding boxes**.
Image classification models are not enough.

Write `{output_dir}/plan/plan.md` with:

1. The article URL and the exact Baidu download source you will use
2. The expected YOLOv8s input size / preprocessing
3. Where the usable inference weights are located after unpacking the archive
4. How YOLO outputs will be converted into `prediction.json`
5. One-image validation plan before batch inference
