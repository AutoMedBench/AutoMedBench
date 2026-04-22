# Sandbox regression tests

Fast, offline tests of the v5 agent sandbox (`agent_code_executor.check_isolation`). No LLM calls, no docker build, no network.

```bash
python eval_image_enhancement/docker/tests/test_sandbox.py
```

Two suites:

| suite | source | what it asserts |
|---|---|---|
| `seg_redteam_corpus` | `seg_redteam_corpus.json` — 75 adversarial snippets authored against the seg sandbox (same 3-layer design v5 ports) | every snippet seg recorded as a bypass is re-checked against v5; regressions are called out |
| `ie_specific` | inline in `test_sandbox.py` | 13 image-enhancement-specific attack vectors (read `reference.npy`, `baseline_bands.json`, `ground_truth.csv`, load `torch.optim`, write `/results`, etc.) must all be blocked; 5 legitimate operations (read `/data/public`, write `/workspace/agents_outputs`, `pip install --user`, …) must all pass |

Last run: 13/13 forbidden blocked, 0 false positives. 16/75 of seg's historical bypasses still get through v5 — documented, not a regression (same layer design, same open weaknesses).

## Scope (what this does NOT test)

- Does not build or launch containers — for that, use a dummy run via `orchestrator.py --dry-run` or the smoke test below.
- Does not exercise the runtime `sys.addaudithook` preamble — that layer only activates when code is actually executed inside a container.
- Does not stress-test an LLM red-teamer. The seg branch has `test_redteam.py` which drives Claude against the sandbox; we didn't port it here because a) it needs an API key, b) it's expensive, c) seg already published the resulting corpus which we re-run above.

## Container smoke

For a full 2-container smoke (no real LLM), run a dummy cell with an agent that's unlikely to produce much:

```bash
# Sets API_KEY to a placeholder; agent container will exit early but the
# orchestrator, mounts, network policy, and eval container all run.
AGENT_API_KEY=dummy python eval_image_enhancement/docker/orchestrator.py \
    --agent claude-opus-4-6 --task ldct-denoising-task --tier lite \
    --n-patients 2 --max-seconds 120 --gpu-id 0 --repeat-idx 0 \
    --output-dir /tmp/ie_smoke_dummy
```
