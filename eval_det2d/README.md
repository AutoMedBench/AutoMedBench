# eval_det2d — 2D Detection Benchmark

Evaluates LLM coding agents on 2D medical object detection tasks. The first task
in this domain is **VinDr-CXR abnormality detection** on chest X-rays.

## Workflow

Agents follow the same S1-S5 workflow pattern as the original benchmark:

1. `S1 PLAN` — research models and write `plan.md`
2. `S2 SETUP` — install dependencies, download weights, load model
3. `S3 VALIDATE` — test on one image, verify box format and quality
4. `S4 INFERENCE` — run all patients, write predictions
5. `S5 SUBMIT` — verify every patient has output and call submit

## Data Layout

```text
data/VinDrCXR_Detection10/
  public/
    VINDR_000001/image.png
  private/
    VINDR_000001/boxes.json
    ground_truth.csv
```

Ground truth format:

```json
{
  "image_id": "03b2...",
  "width": 1024,
  "height": 1024,
  "boxes": [
    {
      "class": "abnormality",
      "source_class": "Lung Opacity",
      "x1": 120.0,
      "y1": 300.0,
      "x2": 240.0,
      "y2": 420.0
    }
  ]
}
```

Agent output format:

```json
{
  "boxes": [
    {
      "class": "abnormality",
      "score": 0.91,
      "x1": 118.0,
      "y1": 302.0,
      "x2": 238.0,
      "y2": 419.0
    }
  ]
}
```

The benchmark is **single-class by default**: all positive findings are mapped
to `abnormality`. The original VinDr-CXR label is preserved in `source_class`
inside the ground truth JSON so the task can later be upgraded to multi-class.

## Scoring

Overall score:

```text
Overall = 50% Workflow + 50% Task Score
Task Score = mAP@0.5
```

For the VinDr-CXR abnormality detection task, this benchmark currently defaults
to **mAP@0.5** to align with recent detection papers that report `mAP@0.5`
and `mAP@0.5:0.95`. If you want to reproduce the older VinBigData competition
setting instead, you can change `iou_threshold` in the task config to `0.4`.

Step weights:

```text
S1 = 25%
S2 = 15%
S3 = 35%
S4 = 15%
S5 = 10%
```

Step definitions:

- `S1`: plan quality and model selection
- `S2`: environment setup, model download, successful load
- `S3`: validation on one image before batch inference
- `S4`: `0.5 * completion_rate + 0.5 * output_format_valid`
- `S5`: `0.5 * has_valid_results + 0.5 * output_format_valid`

Result tiers:

- `good`: `mAP@0.5 >= 0.60`
- `okay`: `0.30 <= mAP@0.5 < 0.60`
- `fail`: `mAP@0.5 < 0.30`

## Preparing a 10-sample test subset

You must first obtain VinDr-CXR yourself from PhysioNet; it is restricted-access.
Then stage a small benchmark subset from the official `test` split:

```bash
python eval_det2d/stage_data.py \
  --source-dir /path/to/vindr-cxr \
  --output-name VinDrCXR_Detection10 \
  --split test \
  --num-samples 10
```

Expected source layout:

```text
/path/to/vindr-cxr/
  test/
    <image_id>.dicom
  annotations_test.csv
```

## Task Folder

The first task lives in:

```text
eval_det2d/vindr-cxr-det-task/
```

and is auto-discovered by `task_loader.py` from `eval_det2d/`.
