# Task Gallery

AutoMedBench currently hosts **13 sub-tasks across 5 domains**. Each branch holds the code, configs, and prompts for one domain.

| Branch | Domain | Sub-task | Source | Task metric |
|---|---|---|---|---|
| [`eval_seg`](../../../tree/eval_seg) | Segmentation | Kidney Tumor | [KiTS19](https://kits19.grand-challenge.org/) | Dice |
| [`eval_seg`](../../../tree/eval_seg) | Segmentation | Pancreas Tumor | [PanTS](https://huggingface.co/datasets/BodyMaps/PanTSMini) | Dice |
| [`eval_seg`](../../../tree/eval_seg) | Segmentation | Pancreas OAR | [PanTS](https://huggingface.co/datasets/BodyMaps/PanTSMini) | macro-Dice |
| [`eval_seg`](../../../tree/eval_seg) | Segmentation | Fetal Brain Tissues | [FeTA](https://fetachallenge.github.io/) | Dice |
| [`eval_image_enhancement`](../../../tree/eval_image_enhancement) | Enhancement | LDCT Denoising | [LDCT-SimNICT](https://www.aapm.org/GrandChallenge/LowDoseCT/) | SSIM |
| [`eval_image_enhancement`](../../../tree/eval_image_enhancement) | Enhancement | MRI Super-Resolution | [SR-MRI](https://www.fastmri.org/) | SSIM |
| [`eval_vqa`](../../../tree/eval_vqa) | VQA | Pathology VQA | [PathVQA](https://github.com/UCSD-AI4H/PathVQA) | accuracy |
| [`eval_vqa`](../../../tree/eval_vqa) | VQA | Radiology VQA | [VQA-RAD](https://osf.io/89kps/) | accuracy |
| [`eval_vqa`](../../../tree/eval_vqa) | VQA | Semantic Radiology VQA | [SLAKE](https://www.med-vqa.com/slake/) | accuracy |
| [`eval_vqa`](../../../tree/eval_vqa) | VQA | Multi-frame Medical VQA | [MedFrameQA](https://huggingface.co/datasets/SuhaoYu1020/MedFrameQA) | accuracy |
| [`eval_vqa`](../../../tree/eval_vqa) | VQA | Expert Multimodal VQA | [MedXpertQA-MM](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | accuracy |
| [`eval_report_gen`](../../../tree/eval_report_gen) | Report Generation | Chest X-ray Report Generation | [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) | MLRG-7 mean |
| [`eval_det2d`](../../../tree/eval_det2d) | Lesion Detection | Chest X-ray Abnormality Detection | [VinDr-CXR](https://vindr.ai/datasets/vindr-cxr) | mAP@0.5 |

Each sub-task directory contains `config.yaml` (task metadata), `model_info.yaml` (candidate models for Standard tier), `requirements.txt`, and per-tier skill markdown files (`lite_s1.md`, `standard_s1.md`, …).
