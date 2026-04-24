# Task Gallery

AutoMedBench currently hosts **14 sub-tasks across 5 domains**. Each branch holds the code, configs, and prompts for one domain.

| Branch | Domain | Sub-task | Source | Task metric |
|---|---|---|---|---|
| [`eval_seg`](../../../tree/eval_seg) | Segmentation | Kidney tumor segmentation | [KiTS19](https://kits19.grand-challenge.org/) | Dice |
| [`eval_seg`](../../../tree/eval_seg) | Segmentation | Liver tumor segmentation | [LiTS](https://competitions.codalab.org/competitions/17094) | Dice |
| [`eval_seg`](../../../tree/eval_seg) | Segmentation | Pancreas tumor segmentation | [PanTS](https://huggingface.co/datasets/BodyMaps/PanTSMini) | Dice |
| [`eval_seg`](../../../tree/eval_seg) | Segmentation | Pancreatic tumor + organs-at-risk (multi-organ) | [AMOS](https://amos22.grand-challenge.org/) | macro-Dice |
| [`eval_seg`](../../../tree/eval_seg) | Segmentation | Fetal brain 7-class segmentation | [FeTA](https://fetachallenge.github.io/) | Dice |
| [`eval_image_enhancement`](../../../tree/eval_image_enhancement) | Enhancement | LDCT denoising | [LDCT-SimNICT](https://www.aapm.org/GrandChallenge/LowDoseCT/) | SSIM |
| [`eval_image_enhancement`](../../../tree/eval_image_enhancement) | Enhancement | MRI super-resolution (×2) | [SR-MRI](https://www.fastmri.org/) | SSIM |
| [`eval_vqa`](../../../tree/eval_vqa) | VQA | Pathology VQA | [PathVQA](https://github.com/UCSD-AI4H/PathVQA) | accuracy |
| [`eval_vqa`](../../../tree/eval_vqa) | VQA | Radiology VQA | [VQA-RAD](https://osf.io/89kps/) | accuracy |
| [`eval_vqa`](../../../tree/eval_vqa) | VQA | Bilingual medical VQA | [SLAKE](https://www.med-vqa.com/slake/) | accuracy |
| [`eval_vqa`](../../../tree/eval_vqa) | VQA | Multi-frame medical VQA | [MedFrameQA](https://huggingface.co/datasets/SuhaoYu1020/MedFrameQA) | accuracy |
| [`eval_vqa`](../../../tree/eval_vqa) | VQA | Expert-level medical VQA | [MedXpertQA-MM](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | accuracy |
| [`eval_report_gen`](../../../tree/eval_report_gen) | Report Generation | CXR findings generation | [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) | MLRG-7 mean |
| [`eval_det2d`](../../../tree/eval_det2d) | Lesion Detection | Chest X-ray abnormality | [VinDr-CXR](https://physionet.org/content/vindr-cxr/) | mAP@0.5 |

Each sub-task directory contains `config.yaml` (task metadata), `model_info.yaml` (candidate models for Standard tier), `requirements.txt`, and per-tier skill markdown files (`lite_s1.md`, `standard_s1.md`, …).
