S3  VALIDATE    Run inference on ONE patient as a sanity check.
  1. Pick one patient and run the full inference pipeline end-to-end.
  2. You MUST use GPU (CUDA) for inference. Verify with torch.cuda.is_available(). \
Load the model onto GPU (e.g. model.cuda() or device='cuda'). If torch.load uses \
map_location, set it to torch.device('cuda'). Only fall back to CPU if CUDA is \
genuinely unavailable — never force CPU when a GPU is present.
  3. Verify the output:
     - Shape matches the original input scan
     - Values are binary (0 and 1 only)
     - Organ mask has a reasonable voxel count (> 1000)
     - CRITICAL — Lesion mask check: compute lesion_voxel_ratio = \
lesion_mask.sum() / organ_mask.sum(). Print this ratio. If it is exactly 0.0, \
your model is NOT detecting tumors — go back to S1 and pick a model with \
actual tumor labels. A working tumor model should produce lesion_voxel_ratio \
between 0.01 and 0.6 on most patients. A ratio of 0.0 means FAILURE.
  4. If the output looks wrong or lesion_voxel_ratio is 0, debug and fix \
the pipeline before proceeding. Do NOT continue to S4 with a model that \
produces empty lesion masks.
  5. If you have been stuck debugging for more than 600 seconds without a \
working pipeline, abandon the current approach and go back to S1 to choose \
a different model or method.
