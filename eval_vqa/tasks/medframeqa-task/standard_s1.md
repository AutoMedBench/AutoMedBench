Choose among exactly these candidates and compare all of them before picking one:

1. `UCSC-VLAA/MedVLThinker-3B-RL_m23k`
2. `MedVLSynther/MedVLSynther-3B-RL_13K`
3. `Qwen/Qwen2.5-VL-3B-Instruct`
4. `google/gemma-4-E2B-it`
5. `google/gemma-4-E4B-it`
6. `microsoft/llava-med-v1.5-mistral-7b`

Planning requirements for `plan.md`:

- name the chosen model explicitly
- list the rejected candidates and why they were not selected
- justify the final choice against MedFrameQA ordered sequence of medical frames + question text + options
- justify one-GPU execution on an A5000
- explain the answer-normalization path — extract one uppercase letter in `A-E` into `predicted_label` and copy the matching option text into `predicted_answer`.
- call out gated-access risk if considering a Gemma 4 candidate
- define the smoke-test path before running a larger slice
