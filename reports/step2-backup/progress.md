# Project progress

## Step 2 completed — 2026-10-20 (Asia/Kolkata)

Existing Mac environment reused. Original reference notebook unchanged.
Working notebook executed successfully with the aml-fashion kernel.
Only official training archives downloaded, with publisher checksum verification.
60,000 grayscale 28x28 uint8 images; raw range 0–255; ten balanced classes.
Saved seed-42 stratified indices: 48,000 training and 12,000 validation;
4,800 and 1,200 examples respectively per class, no shared row indices.
Checked complete coverage, pixel scaling to float32 [0,1], and identical split reload.
Saved preprocessing contract, class counts, representative training examples, and summary.
No model trained. Official test files neither downloaded nor loaded.

Next milestone: Step 3 baselines, with the same saved split. Do not reinstall the environment.
