# Dataset Collection

This page tracks the dataset sources used by AutoMedBench and the release
policy for the public benchmark packages. It is a convenience index, not legal
advice and not a license grant.

## Live releases

| Release | Link | Contents | Dataset handling |
|---|---|---|---|
| AutoMedBench-Lite-v0.1 | [Hugging Face release](https://huggingface.co/datasets/MitakaKuma/AutoMedBench-Lite-release) | 7 Lite held-out sandbox tasks | Includes staged Lite subsets for local scoring; see the release `DATA_CARD.md` and obey each upstream source term. |
| AutoMedBench-Full-v0.1 | [Hugging Face release](https://huggingface.co/datasets/MitakaKuma/AutoMedBench-Full-release) | 48 tasks across 7 tracks; Lite and Standard tiers; 96 task-tier combinations | Dataset bytes are not bundled. Users must download authorized source data independently and mount it at runtime. |

## Data layout

Before each run, the staging code materializes a public/private split:

```text
data/<DatasetName>/
  public/<patient_id>/     # inputs the agent sees
  private/<patient_id>/    # ground truth the eval container scores against
```

The agent container has no mount to `private/`. The eval container runs with
`--network none` and scores agent outputs against `private/` offline. For
restricted datasets, the runner must hold valid credentials or an accepted data
use agreement before staging local files.

## Licensing and access policy

AutoMedBench code is MIT-licensed under this repository's `LICENSE`. That code
license does not apply to any upstream dataset.

The table below records the current public source, license or access term, and
release readiness status used by AutoMedBench-Full-v0.1. It should be read
conservatively:

- If a row says `Restricted`, `Credentialed DUA`, `Controlled`,
  `Permission required`, `Mixed`, `No explicit license`, `No explicit
  redistribution grant`, `No clear bulk redistribution license`, or
  `Per-image`, do not redistribute the data unless you have separate written
  permission or a governing agreement that allows it.
- If a row lists a permissive or Creative Commons license, attribution and all
  license conditions still apply.
- Always verify the current upstream page before using a dataset for a new
  purpose, especially for commercial use or redistribution.

## Full-v0.1 task source matrix

| Track | Task ID | Upstream dataset | Source | License / terms | Access / status |
|---|---|---|---|---|---|
| Segmentation | `kidney-seg-task` | KiTS19 | [kits19.grand-challenge.org](https://kits19.grand-challenge.org/) | CC BY-NC-SA 4.0 | Public; acquisition ready |
| Segmentation | `pancreas-seg-task` | PanTS | [github.com/MrGiovanni/PanTS](https://github.com/MrGiovanni/PanTS) | CC BY-NC-ND 4.0 | Public; acquisition ready |
| Segmentation | `pancreas-oar-seg-task` | PanTS | [github.com/MrGiovanni/PanTS](https://github.com/MrGiovanni/PanTS) | CC BY-NC-ND 4.0 | Public; acquisition ready |
| Segmentation | `liver-seg-task` | MSD Task03 Liver | [medicaldecathlon.com](https://medicaldecathlon.com/) | Provenance conflict | Public; staged verified |
| Segmentation | `aeropath-seg-task` | AeroPath | [zenodo.org/records/10069289](https://zenodo.org/records/10069289) | CC BY 4.0 | Public; acquisition ready |
| Segmentation | `tsg-multiorgan-seg-task` | TotalSegmentator v2.0.1 | [zenodo.org/records/10047263](https://zenodo.org/records/10047263) | CC BY 4.0 | Public; staged verified |
| Segmentation | `colon-seg-task` | MSD Task10 Colon | [medicaldecathlon.com](https://medicaldecathlon.com/) | CC BY-SA 4.0 | Public; staged verified |
| Segmentation | `hepaticvessel-seg-task` | MSD Task08 HepaticVessel | [medicaldecathlon.com](https://medicaldecathlon.com/) | CC BY-SA 4.0 | Public; staged verified |
| Segmentation | `spleen-seg-task` | MSD Task09 Spleen | [medicaldecathlon.com](https://medicaldecathlon.com/) | CC BY-SA 4.0 | Public; staged verified |
| Segmentation | `heart-seg-task` | MSD Task02 Heart | [medicaldecathlon.com](https://medicaldecathlon.com/) | CC BY-SA 4.0 | Public; staged verified |
| Segmentation | `prostate-seg-task` | MSD Task05 Prostate | [medicaldecathlon.com](https://medicaldecathlon.com/) | CC BY-SA 4.0 | Public; staged verified |
| Segmentation | `feta-seg-task` | FeTA | [zenodo.org/records/4541606](https://zenodo.org/records/4541606) | Research agreement | Gated; externally blocked |
| Segmentation | `panther-t1-seg-task` | PANTHER | [zenodo.org/records/15192302](https://zenodo.org/records/15192302) | CC BY-NC 4.0 + access controls | Restricted; externally blocked |
| Segmentation | `panther-t2-seg-task` | PANTHER | [zenodo.org/records/15192302](https://zenodo.org/records/15192302) | CC BY-NC 4.0 + access controls | Restricted; externally blocked |
| Enhancement | `ldct-denoising-task` | AAPM Low Dose CT Grand Challenge | [aapm.org/grandchallenge/lowdosect](https://www.aapm.org/GrandChallenge/LowDoseCT/) | No explicit redistribution grant | Public library; download required |
| Enhancement | `lidc-idri-denoising-task` | LIDC-IDRI | [cancerimagingarchive.net/collection/lidc-idri](https://www.cancerimagingarchive.net/collection/lidc-idri/) | CC BY 3.0 | Public; acquisition ready |
| Enhancement | `deeplesion-denoising-task` | NIH DeepLesion | [nihcc.app.box.com/v/DeepLesion](https://nihcc.app.box.com/v/DeepLesion) | No explicit license | Public; acquisition ready |
| Enhancement | `mri-sr-task` | fastMRI | [fastmri.med.nyu.edu](https://fastmri.med.nyu.edu/) | Agreement prohibits redistribution | Application required; externally blocked |
| Enhancement | `brats-t1c-sr-task` | BraTS 2023 GLI | [synapse.org/Synapse:syn51156910](https://www.synapse.org/Synapse:syn51156910/wiki/) | Controlled access | Controlled; externally blocked |
| Enhancement | `ixi-t1-sr-task` | IXI T1 | [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/) | CC BY-SA 3.0 | Public; acquisition ready |
| Enhancement | `nih-cxr-sr-task` | NIH ChestXray14 | [nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC) | CC0 (NIH Kaggle) | Public; acquisition ready |
| VQA | `pathvqa-task` | PathVQA | [huggingface.co/datasets/flaviagiammarino/path-vqa](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | MIT | Public; staged verified |
| VQA | `vqa-rad-task` | VQA-RAD | [huggingface.co/datasets/flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | CC0 1.0 | Public; staged verified |
| VQA | `medframeqa-task` | MedFrameQA | [huggingface.co/datasets/SuhaoYu1020/MedFrameQA](https://huggingface.co/datasets/SuhaoYu1020/MedFrameQA) | CC BY 4.0 | Public; staged verified |
| VQA | `slake-task` | SLAKE | [huggingface.co/datasets/BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) | CC BY 4.0 | Public; staged verified |
| VQA | `medxpertqa-mm-task` | MedXpertQA | [huggingface.co/datasets/TsinghuaC3I/MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | MIT | Public; staged verified |
| VQA | `vqa-kvasir-task` | Kvasir-VQA | [huggingface.co/datasets/SimulaMet-HOST/Kvasir-VQA](https://huggingface.co/datasets/SimulaMet-HOST/Kvasir-VQA) | CC BY-NC 4.0 + benchmark permission | Permission required; externally blocked |
| VQA | `vqa-omnimedvqa-task` | OmniMedVQA | [huggingface.co/datasets/foreverbeliever/OmniMedVQA](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA) | Mixed; no global license | Mixed access; externally blocked |
| VQA | `vqa-pmc-vqa-task` | PMC-VQA | [huggingface.co/datasets/RadGenome/PMC-VQA](https://huggingface.co/datasets/RadGenome/PMC-VQA) | CC BY-SA | Public; staged verified |
| VQA | `vqa-mmmu-medical-task` | MMMU medical subsets | [huggingface.co/datasets/MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU) | Apache-2.0 | Public; staged verified |
| Report generation | `mimic-cxr-report-task` | MIMIC-CXR | [physionet.org/content/mimic-cxr/2.1.0](https://physionet.org/content/mimic-cxr/2.1.0/) | DUA prohibits sharing | Credentialed DUA; externally blocked |
| Report generation | `iu-xray-report-task` | IU Open-i | [openi.nlm.nih.gov](https://openi.nlm.nih.gov/) | No clear bulk redistribution license | Public search; acquisition ready |
| Report generation | `chexpert-plus-cxr-task` | CheXpert Plus | [aimi.stanford.edu/datasets/chexpert-plus](https://aimi.stanford.edu/datasets/chexpert-plus) | No license declared by mirror | Public mirror; staged verified |
| Report generation | `pathology-caption-100-task` | PathCap | [huggingface.co/datasets/jamessyx/PathCap](https://huggingface.co/datasets/jamessyx/PathCap) | CC BY-NC 2.0 + click-through | Auto-gated; staged verified |
| Report generation | `pathology-caption-500-task` | PathCap | [huggingface.co/datasets/jamessyx/PathCap](https://huggingface.co/datasets/jamessyx/PathCap) | CC BY-NC 2.0 + click-through | Auto-gated; staged verified |
| Detection | `vindr-cxr-det-task` | VinDr-CXR | [physionet.org/content/vindr-cxr/1.0.0](https://physionet.org/content/vindr-cxr/1.0.0/) | PhysioNet Credentialed Health Data License | Credentialed DUA; externally blocked |
| Detection | `bccd-det-task` | BCCD | [github.com/Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset) | MIT | Public; staged verified |
| Detection | `dentex-det-task` | DENTEX | [huggingface.co/datasets/ibrahimhamamci/DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) | CC BY-NC-SA 4.0 | Public; staged verified |
| Detection | `grazpedwri-det-task` | GRAZPEDWRI-DX | [figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193) | CC BY 4.0 | Public; staged verified |
| Classification | `braintumor-cls-task` | Brain Tumor MRI | [kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | CC BY 4.0 | Public; acquisition ready |
| Classification | `crc-histology-cls-task` | NCT-CRC-HE-100K | [zenodo.org/records/1214456](https://zenodo.org/records/1214456) | CC BY 4.0 | Public; acquisition ready |
| Classification | `patchcamelyon-cls-task` | PatchCamelyon | [github.com/basveeling/pcam](https://github.com/basveeling/pcam) | CC0 | Public; acquisition ready |
| Classification | `chest-xray-pneumonia-cls-task` | Kermany chest X-ray | [data.mendeley.com/datasets/rscbjbr9sj/2](https://data.mendeley.com/datasets/rscbjbr9sj/2) | CC BY 4.0 | Public; acquisition ready |
| Classification | `skin-lesion-cls-task` | ISIC Archive | [isic-archive.com](https://www.isic-archive.com/) | Per-image; not frozen | Public; staged verified |
| Synthesis | `synthrad2025-mrct-task` | SynthRAD2025 | [zenodo.org/records/15373853](https://zenodo.org/records/15373853) | CC BY-NC 4.0 | Public; acquisition ready |
| Synthesis | `ctorg-ctsr-task` | CT-ORG | [cancerimagingarchive.net/collection/ct-org](https://www.cancerimagingarchive.net/collection/ct-org/) | CC BY 3.0 | Public; acquisition ready |
| Synthesis | `msd-pancreas-ctsr-task` | MSD Task07 Pancreas | [medicaldecathlon.com](https://medicaldecathlon.com/) | CC BY-SA 4.0 | Public; staged verified |
| Synthesis | `totalsegmentator-ctsr-task` | TotalSegmentator | [github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | CC BY 4.0 | Public; acquisition ready |

## Lite-v0.1 packaged task sources

AutoMedBench-Lite-v0.1 is a smaller local sandbox release. It packages staged
Lite subsets for offline local scoring, so review the release data card and
upstream terms before redistribution or use outside the intended benchmark
workflow.

| Track | Task ID | Upstream source | License / terms |
|---|---|---|---|
| Classification | `skin-lesion-cls-task` | [HAM10000 / ISIC 2018](https://challenge.isic-archive.com/data/) | CC BY-NC 4.0 |
| Synthesis | `msd-pancreas-ctsr-task` | [Medical Segmentation Decathlon Task07 Pancreas](http://medicaldecathlon.com/) / [AWS Open Data mirror](https://registry.opendata.aws/msd/) | CC BY-SA 4.0 |
| Detection | `grazpedwri-det-task` | [GRAZPEDWRI-DX](https://www.nature.com/articles/s41597-022-01328-z) | CC BY 4.0 |
| Segmentation | `tsg-multiorgan-seg-task` | [TotalSegmentator CT-Lite](https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite) | CC BY 4.0 |
| VQA | `medxpertqa-mm-task` | [MedXpertQA-MM](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | MIT |
| Report generation | `chexpert-plus-cxr-task` | [CheXpert-plus-RRG](https://huggingface.co/datasets/X-iZhang/CheXpert-plus-RRG), derived from [CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus) | License not declared on the mirror; follow CheXpert Plus / Stanford source terms |
| Enhancement | `ldct-denoising-task` | [SimNICT](https://huggingface.co/datasets/YutingHe-list/SimNICT) | CC BY-ND 4.0 |

## Strict-access and unclear-license datasets

The following sources require special care before use:

- **Credentialed or gated access:** MIMIC-CXR, VinDr-CXR, FeTA, PANTHER,
  fastMRI, BraTS 2023 GLI, and Kvasir-VQA.
- **Mixed or unclear redistribution terms:** AAPM LDCT, NIH DeepLesion,
  IU/Open-i, CheXpert Plus mirrors, OmniMedVQA, and ISIC per-image subsets.
- **Non-commercial or no-derivatives terms:** CC BY-NC, CC BY-NC-SA,
  CC BY-NC-ND, and CC BY-ND rows should be treated as incompatible with
  commercial redistribution unless the upstream owner grants explicit
  permission.

For the Full release, the conservative rule is simple: no dataset bytes are
embedded in any task package, runtime image, or task-tier image. Runtime data is
mounted from a local directory prepared independently by the authorized user.
