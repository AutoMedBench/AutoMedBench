# Dataset Collection

## Datasets

| Dataset | Task | Challenge / Paper | Year |
|---|---|---|---|
| KiTS19 | Segmentation | [KiTS19 Challenge](https://kits19.grand-challenge.org/) | 2019 |
| LiTS | Segmentation | [LiTS Challenge](https://competitions.codalab.org/competitions/17094) | 2017 |
| PanTS | Segmentation | [BodyMaps PanTS](https://huggingface.co/datasets/BodyMaps/PanTSMini) | 2024 |
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

## Licensing & access

> **Important.** AutoMedBench does not redistribute any dataset. Each dataset is governed by its own license and access requirements — the table below is a convenience pointer, not a legal statement. **Always verify the current license on the official source page** before using any dataset, and consult your institution's legal/compliance office for any use beyond individual non-commercial research.

| Dataset | License | Access | Commercial use | Redistribution |
|---|---|---|---|---|
| KiTS19 | see [source](https://github.com/neheller/kits19) | public download | verify on source | verify on source |
| LiTS | verify on source | public (Codalab) | verify on source | verify on source |
| PanTS | [CC BY-NC-SA 4.0](https://huggingface.co/datasets/BodyMaps/PanTSMini) | public (HuggingFace) | **not permitted** | permitted under same license |
| FeTA | verify on source | request-based | verify on source | verify on source |
| LDCT-SimNICT | verify on source ([AAPM](https://www.aapm.org/GrandChallenge/LowDoseCT/)) | research-agreement | verify on source | verify on source |
| SR-MRI (fastMRI) | [NYU Langone Data Sharing Agreement](https://fastmri.med.nyu.edu/) | application required | **not permitted** | **not permitted** |
| PathVQA | see source | public (GitHub) | verify on source | verify on source |
| VQA-RAD | verify on source | public (OSF) | verify on source | verify on source |
| SLAKE | verify on source | public download | verify on source | verify on source |
| MedFrameQA | see HuggingFace card | public (HuggingFace) | verify on source | verify on source |
| MedXpertQA-MM | see HuggingFace card | public (HuggingFace) | verify on source | verify on source |
| MIMIC-CXR | [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimic-cxr/) | **credentialed; DUA + CITI training required** | **not permitted** | **not permitted** |
| VinDr-CXR | [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/vindr-cxr/) | **credentialed; DUA + CITI training required** | **not permitted** | **not permitted** |

### Strict-access datasets

Three datasets require institutional credentialing and a signed Data Use Agreement before anyone may access the files. Running the benchmark tasks that depend on them requires the runner to hold valid credentials. AutoMedBench never stages this data into any public branch.

- **MIMIC-CXR** — PhysioNet credentialed; DUA + CITI training. No redistribution, no commercial use.
- **VinDr-CXR** — same PhysioNet credentialed license. No redistribution, no commercial use.
- **SR-MRI (fastMRI)** — NYU Langone Data Sharing Agreement; application required. No redistribution, no commercial use; MLCommons/MLPerf benchmarking is the only carve-out.

### A note on AutoMedBench's own license

The AutoMedBench **code** is MIT-licensed (see `LICENSE`). The code license does not extend to any dataset it reads: using AutoMedBench to run tasks on a given dataset means you are also bound by that dataset's license and access terms.
