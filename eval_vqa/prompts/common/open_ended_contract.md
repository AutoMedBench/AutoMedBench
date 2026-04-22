Open-ended answer contract (this task uses `answer_mode: open_ended`):
- `predicted_answer` must be a **short span of ≤ 5 words** (typically one word or a short noun phrase); VLM prose must be reduced to that span by `answer_postprocess.py`.
- Yes/no questions: write exactly `yes` or `no`, nothing else.
- No leading articles (`the`/`a`/`an`), trailing punctuation, or explanations.
- Plan S1 around building a postprocess that extracts this short-answer span; validate in S2 smoke that `postprocess(raw_model_output)` already returns ≤ 5 words on your smoke sample before moving to S3.
- Scoring uses LLM-as-judge + `0.5·EM + 0.5·token_F1`; both collapse to ~0 on long sentences even if the answer span is present.
