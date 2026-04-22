S1  PLAN        This is a head-to-head competition against other AI agents.
  Your final score depends entirely on the quality of the enhancement method \
you pick. Every other agent is solving this exact same task right now. The \
agent that finds the best inference-only method wins. Settling for the first \
classical baseline guarantees a mediocre score.

  Your objective: find the absolute best inference-only enhancement method for \
{modality} {task_type}.

  Rules:
  - Do NOT rely on what you think you know. Search actively. Model \
repositories, checkpoints, and packages change constantly.
  - Do NOT stop at the first method that works. You MUST find and evaluate at \
least 3 distinct candidates before choosing.
  - Search broadly: HuggingFace, GitHub, MONAI Model Zoo, `pip search` equivalents, \
challenge leaderboards, project pages, paper with code.
  - INFERENCE-ONLY is a hard constraint. Methods that require training on the \
target data are OUT OF SCOPE. Methods with public pretrained checkpoints or \
classical analytical formulations are IN SCOPE.
  - Over-smoothing hurts LPIPS and SSIM. A method that produces crisp \
structure-preserving output beats one that blurs everything.

  Steps:
  1. Conduct wide research. Do not download anything yet — just catalog what \
exists, whether pretrained weights are public, and how each method expects its \
input. Be efficient — spend no more than 10-15 tool calls on research.
  2. For every candidate, verify programmatically: (a) weights (or package) are \
actually obtainable in this environment, (b) the input format is compatible \
with {modality} data in this dataset, (c) no training is required.
  3. Rank candidates by expected enhancement quality. Pick the best.
  4. Save your plan to {output_dir}/plan/plan.md — include every candidate \
found, full comparison, final decision with justification, and your complete \
pipeline design.
  5. Generate a flowchart as {output_dir}/plan/plan.png.
