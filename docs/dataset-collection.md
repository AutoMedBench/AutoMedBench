# Dataset Collection

## Datasets

| Dataset | Task | Challenge / Paper | Year |
|---|---|---|---|
| KiTS19 | Segmentation | [KiTS19 Challenge](https://kits19.grand-challenge.org/) | 2019 |
| PanTS | Segmentation | [BodyMaps PanTS](https://huggingface.co/datasets/BodyMaps/PanTSMini) | 2024 |
| FeTA | Segmentation | [FeTA Challenge](https://fetachallenge.github.io/) | 2021 |
| AeroPath | Segmentation | [AeroPath](https://github.com/raidionics/AeroPath) | 2023 |
| TotalSegmentator | Segmentation | [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | 2023 |
| PANTHER 2025 (T1, T2) | Segmentation | [PANTHER Challenge](https://panther.grand-challenge.org/) | 2025 |
| LDCT-SimNICT | Enhancement | [AAPM Low-Dose CT Grand Challenge](https://www.aapm.org/GrandChallenge/LowDoseCT/) | 2016 |
| SR-MRI | Enhancement | [fastMRI](https://www.fastmri.org/) | 2018 |
| PathVQA | VQA | [PathVQA (HuggingFace)](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | 2020 |
| VQA-RAD | VQA | [VQA-RAD (HuggingFace)](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | 2018 |
| MedFrameQA | VQA | [MedFrameQA (HuggingFace)](https://huggingface.co/datasets/SuhaoYu1020/MedFrameQA) | 2025 |
| SLAKE-EN | VQA | [SLAKE-EN (HuggingFace)](https://huggingface.co/datasets/BoKelvin/SLAKE) | 2021 |
| MedXpertQA-MM | VQA | [TsinghuaC3I MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | 2024 |
| MIMIC-CXR | Report Generation | [PhysioNet MIMIC-CXR](https://physionet.org/content/mimic-cxr/) | 2019 |
| IU / Open-i | Report Generation | [Open-i (NLM)](https://openi.nlm.nih.gov/) | 2015 |
| CheXpert Plus | Report Generation | [Stanford AIMI CheXpert Plus](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1) | 2024 |
| PathCap | Report Generation | [PathCap (HuggingFace)](https://huggingface.co/datasets/jamessyx/PathCap) | 2024 |
| VinDr-CXR | Lesion Detection | [VinDr.ai VinDr-CXR](https://vindr.ai/datasets/vindr-cxr) | 2022 |
| BCCD | Lesion Detection | [BCCD (HuggingFace)](https://huggingface.co/datasets/keremberke/blood-cell-object-detection) | 2018 |
| DENTEX | Lesion Detection | [DENTEX (HuggingFace)](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) | 2023 |
| GRAZPEDWRI-DX | Lesion Detection | [GRAZPEDWRI-DX (figshare)](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193) | 2022 |

Every dataset satisfies three requirements: (1) publicly available, (2) deterministic ground truth, (3) peer-reviewed / challenge-backed.

## Data layout

Before each run, `stage_data.py` materializes a public/private split:

```
data/<DatasetName>/
  public/<patient_id>/     # inputs the agent sees
  private/<patient_id>/    # ground truth the eval container scores against
```

The agent container has **no mount** to `private/`. The eval container runs with `--network none` and scores agent outputs against `private/` offline.

## Licensing & access

> **Important.** AutoMedBench does not redistribute any dataset. Each dataset is governed by its own license and access requirements — the table below is a convenience pointer, not a legal statement. **Always verify the current license on the official source page** before using any dataset, and consult your institution's legal/compliance office for any use beyond individual non-commercial research.

| Dataset | License | Access | Commercial use | Redistribution |
|---|---|---|---|---|
| KiTS19 | see [source](https://github.com/neheller/kits19) | public download | verify on source | verify on source |
| PanTS | [CC BY-NC-SA 4.0](https://huggingface.co/datasets/BodyMaps/PanTSMini) | public (HuggingFace) | **not permitted** | permitted under same license |
| FeTA | verify on source | request-based | verify on source | verify on source |
| AeroPath | see [source](https://github.com/raidionics/AeroPath) | public download | verify on source | verify on source |
| TotalSegmentator | see [source](https://github.com/wasserth/TotalSegmentator) | public download | verify on source | verify on source |
| PANTHER 2025 (T1, T2) | see [challenge terms](https://panther.grand-challenge.org/) | challenge registration | verify on source | verify on source |
| LDCT-SimNICT | verify on source ([AAPM](https://www.aapm.org/GrandChallenge/LowDoseCT/)) | research-agreement | verify on source | verify on source |
| SR-MRI (fastMRI) | [NYU Langone Data Sharing Agreement](https://fastmri.med.nyu.edu/) | application required | **not permitted** | **not permitted** |
| PathVQA | see [HuggingFace card](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | public (HuggingFace) | verify on source | verify on source |
| VQA-RAD | see [HuggingFace card](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | public (HuggingFace) | verify on source | verify on source |
| MedFrameQA | see [HuggingFace card](https://huggingface.co/datasets/SuhaoYu1020/MedFrameQA) | public (HuggingFace) | verify on source | verify on source |
| SLAKE-EN | see [HuggingFace card](https://huggingface.co/datasets/BoKelvin/SLAKE) | public (HuggingFace) | verify on source | verify on source |
| MedXpertQA-MM | see HuggingFace card | public (HuggingFace) | verify on source | verify on source |
| MIMIC-CXR | [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimic-cxr/) | **credentialed; DUA + CITI training required** | **not permitted** | **not permitted** |
| IU / Open-i | see [Open-i terms](https://openi.nlm.nih.gov/) | public (NLM) | verify on source | verify on source |
| CheXpert Plus | see [Stanford AIMI license](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1) | research registration (Stanford AIMI) | verify on source | verify on source |
| PathCap | see [HuggingFace card](https://huggingface.co/datasets/jamessyx/PathCap) | public (HuggingFace) | verify on source | verify on source |
| VinDr-CXR | see [VinDr.ai terms](https://vindr.ai/datasets/vindr-cxr) | public (VinDr.ai) | verify on source | verify on source |
| BCCD | see [HuggingFace card](https://huggingface.co/datasets/keremberke/blood-cell-object-detection) | public (HuggingFace) | verify on source | verify on source |
| DENTEX | see [HuggingFace card](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) | public (HuggingFace) | verify on source | verify on source |
| GRAZPEDWRI-DX | see [figshare record](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193) | public (figshare) | verify on source | verify on source |

### Strict-access datasets

Two datasets require institutional credentialing and a signed Data Use Agreement before anyone may access the files. Running the benchmark tasks that depend on them requires the runner to hold valid credentials. AutoMedBench never stages this data into any public branch.

- **MIMIC-CXR** — PhysioNet credentialed; DUA + CITI training. No redistribution, no commercial use.
- **SR-MRI (fastMRI)** — NYU Langone Data Sharing Agreement; application required. No redistribution, no commercial use; MLCommons/MLPerf benchmarking is the only carve-out.

### A note on AutoMedBench's own license

The AutoMedBench **code** is MIT-licensed (see `LICENSE`). The code license does not extend to any dataset it reads: using AutoMedBench to run tasks on a given dataset means you are also bound by that dataset's license and access terms.
