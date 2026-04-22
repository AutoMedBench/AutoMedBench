## Lite S3 Skill

Before batch inference:

1. Run exactly one image first with the article's YOLOv8s checkpoint
2. Save a sample `prediction.json`
3. Check every box has `class`, `score`, `x1`, `y1`, `x2`, `y2`
4. Confirm YOLO coordinates are mapped back to the original image bounds
5. If the image should be positive, make sure the model is not returning an empty list
