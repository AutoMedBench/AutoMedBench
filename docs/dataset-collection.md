# Dataset Collection

## Datasets

| Dataset | Task | Challenge / Paper | Year |
|---|---|---|---|
| KiTS19 | Segmentation | [KiTS19 Challenge](https://kits19.grand-challenge.org/) | 2019 |
| LiTS | Segmentation | [LiTS Challenge](https://competitions.codalab.org/competitions/17094) | 2017 |
| PanTS | Segmentation | [BodyMaps PanTS](https://huggingface.co/datasets/BodyMaps/PanTSMini) | 2024 |
| AMOS | Segmentation | [AMOS Challenge](https://amos22.grand-challenge.org/) | 2022 |
| FeTA | Segmentation | [FeTA Challenge](https://fetachallenge.github.io/) | 2021 |
| LDCT-SimNICT | Enhancement | [AAPM Low-Dose CT Grand Challenge](https://www.aapm.org/GrandChallenge/LowDoseCT/) | 2016 |
| SR-MRI | Enhancement | [fastMRI](https://www.fastmri.org/) | 2018 |
| PathVQA | VQA | [He et al., 2020](https://arxiv.org/abs/2003.10286) | 2020 |
| VQA-RAD | VQA | [Lau et al., 2018](https://osf.io/89kps/) | 2018 |
| SLAKE | VQA | [Liu et al., ISBI 2021](https://www.med-vqa.com/slake/) | 2021 |
| MedFrameQA | VQA | [HuggingFace dataset](https://huggingface.co/datasets/SuhaoYu1020/MedFrameQA) | 2024 |
| MedXpertQA-MM | VQA | [TsinghuaC3I MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | 2024 |
| MIMIC-CXR | Report Generation | [PhysioNet MIMIC-CXR](https://physionet.org/content/mimic-cxr/) | 2019 |
| VinDr-CXR | Lesion Detection | [PhysioNet VinDr-CXR](https://physionet.org/content/vindr-cxr/) | 2022 |

Every dataset satisfies three requirements: (1) publicly available, (2) deterministic ground truth, (3) peer-reviewed / challenge-backed.

## Data layout

Before each run, `stage_data.py` materializes a public/private split:

```
data/<DatasetName>/
  public/<patient_id>/     # inputs the agent sees
  private/<patient_id>/    # ground truth the eval container scores against
```

The agent container has **no mount** to `private/`. The eval container runs with `--network none` and scores agent outputs against `private/` offline.

## Licensing

Each dataset keeps its original license. Redistribution of any AutoMedBench artifact that touches a dataset must preserve that dataset's license terms (KiTS, LiTS, MSD, FeTA, MIMIC-CXR, PathVQA, VQA-RAD, SLAKE, VinDr-CXR, etc.).
