S1  PLAN        This is a head-to-head competition against other AI agents.
  Your final score depends entirely on the quality of the model(s) you find. \
Every other agent is solving this exact same task right now. The agent that \
finds the best model wins. Settling for the first working model guarantees \
a low ranking.

  Your objective: find the absolute best pre-trained model(s) for {organ} \
organ and TUMOR segmentation in {modality} scans.

  Rules:
  - Do NOT rely on what you think you know. Search actively. Your training \
data is stale — models, checkpoints, and repositories change constantly.
  - Do NOT stop at the first model that works. You MUST find and evaluate \
at least 3 distinct candidates before choosing. Agents that skip \
research and jump to inference will be scored harshly.
  - Search broadly: HuggingFace, GitHub, MONAI Model Zoo, package \
repositories, challenge leaderboards, project pages. Any source is valid.
  - TUMOR detection is what separates winning agents from losing ones. \
A model that segments the organ but misses tumors will score near zero \
on the most heavily weighted metric (lesion Dice).
  - You need a model trained specifically on tumor-annotated data. \
General-purpose organ segmentation tools do NOT have tumor labels.

  Steps:
  1. Conduct a wide research across available sources. Download \
nothing yet — just catalog what exists, what labels each model covers, \
and where the weights live. Be efficient — spend no more than 10-15 \
tool calls on research before making a decision.
  2. For every candidate, verify programmatically: (a) weights are actually \
downloadable, (b) label map covers BOTH {organ} organ AND {organ} \
lesions/tumors, (c) compatible with the {modality} data in this dataset.
  3. If no single model covers both, design a multi-model pipeline.
  4. Rank all candidates by expected lesion segmentation quality. Pick the \
best. Justify with hard evidence — Dice scores, challenge rankings, \
label coverage.
  5. Save your plan to {output_dir}/plan/plan.md — include: every candidate \
found, full comparison, final decision with justification, and your \
complete pipeline design.
  6. Generate a flowchart of your pipeline as {output_dir}/plan/plan.png.
