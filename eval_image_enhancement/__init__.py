"""eval_image_enhancement — MedAgentsBench image-enhancement domain.

Sibling of eval_seg/. Evaluates coding agents on 2D medical image enhancement
tasks (LDCT denoising, MRI super-resolution). PET denoising cut from v1.

Scoring: PSNR + SSIM + LPIPS against a private hidden test set with per-case
randomized degradation. No downstream clinical tasks in v1.
"""
