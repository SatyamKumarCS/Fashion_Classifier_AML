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

## Step 3 completed — 2026-10-20 (Asia/Kolkata)

Trained 16 predeclared candidate configurations: pixels/HOG × Logistic Regression,
kNN, Random Forest, linear SVM × two settings. Shared seed-42 stratified subset
of 12,000 training images, 1,200 per class; full 12,000-image validation split.
Selected and saved eight models by validation macro F1 (accuracy tie breaker),
with training-only scalers inside LR/SVM pipelines. Reloaded all eight models
and verified predictions against saved validation outputs. Original shared split
and reference notebook unchanged. Official test data untouched.

Best validation configuration: pixels/random_forest;
accuracy 0.864583, macro F1 0.862814.
These are budgeted validation results, not official test scores.
Next: Step 4 CNN. Document or align training budgets before final comparison.
