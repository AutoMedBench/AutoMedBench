# Dataset Collection

Every dataset in AutoMedBench satisfies three requirements:

1. **Publicly available** — no internal-only data.
2. **Deterministic ground truth** — binary masks, exact labels, bounding boxes. No free-text judgments.
3. **Peer-reviewed benchmark datasets** — established in the medical imaging literature so results are comparable.

## Datasets currently used

| Domain | Datasets |
|---|---|
| Segmentation | KiTS19 · LiTS · MSD-Pancreas · PanTS · FeTA |
| Image enhancement | LDCT-SimNICT · SR-MRI |
| Report generation | MIMIC-CXR |
| VQA | PathVQA · VQA-RAD · SLAKE · MedFrameQA · MedXpertQA-MM |
| Detection | VinDr-CXR |

## Data layout

Before each run, `stage_data.py` materializes a public/private split:

```
data/<DatasetName>/
  public/<patient_id>/     # inputs the agent sees
  private/<patient_id>/    # ground truth the eval container scores against
```

The agent container has **no mount** to `private/`. The eval container runs with `--network none` and scores agent outputs against `private/` offline.

## Licensing

Each dataset keeps its original license. Redistribution of any AutoMedBench artifact that touches a dataset must preserve that dataset's license terms (KiTS, LiTS, MSD, FeTA, MIMIC-CXR, etc.).
