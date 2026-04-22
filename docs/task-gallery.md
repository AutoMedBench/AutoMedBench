# Task Gallery

AutoMedBench currently hosts **13 tasks across 5 domains**. Each task is defined on its own branch.

| Branch | Domain | Tasks | Clinical metric |
|---|---|---|---|
| [`eval_seg`](../../../tree/eval_seg) | 3D segmentation | kidney · liver · pancreas · feta (7-class fetal brain) | mean Dice |
| [`eval_image_enhancement`](../../../tree/eval_image_enhancement) | 2D image enhancement | LDCT denoising · MRI ×2 super-resolution | mean SSIM |
| [`eval_report_gen`](../../../tree/eval_report_gen) | CXR report generation | MIMIC-CXR findings | MLRG 7-metric mean |
| [`eval_vqa`](../../../tree/eval_vqa) | medical VQA | PathVQA · VQA-RAD · SLAKE · MedFrameQA · MedXpertQA-MM | accuracy |
| [`eval_det2d`](../../../tree/eval_det2d) | 2D detection *(beta)* | VinDr-CXR abnormality | mAP@0.5 |

Each task directory contains `config.yaml` (task metadata), `model_info.yaml` (candidate models for Standard tier), `requirements.txt`, and per-tier skill markdown files (`lite_s1.md`, `standard_s1.md`, …).
