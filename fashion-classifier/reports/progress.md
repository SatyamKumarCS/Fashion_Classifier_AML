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

## Step 4 completed — 2026-10-20 (Asia/Kolkata)

Trained a two-block CNN with 429,450 trainable parameters on mps.
Exactly the saved 12,000 baseline training indices and 12,000 validation indices.
Adam lr 0.001, weight decay 0.0001, batch 128, dropout 0.4, seed 42,
max 25 epochs and five-epoch validation-loss patience.
Epochs run: 14; best validation-loss epoch: 9;
early stopping triggered: True. Restored best checkpoint:
validation accuracy 0.899250, macro F1 0.898020.
Saved portable CPU-tensor weights, metadata, learning history, predictions and curves.
Verified CPU reload against saved probabilities and recomputed metrics.
Protected-file hashes confirm reference, shared split, subset and baseline models unchanged.
No official test downloaded or loaded. These are single-seed validation scores.
Next: Step 5 final evaluation after freezing model choices.

## Step 5 completed — 2026-10-20 (Asia/Kolkata)

Before official test access, saved frozen-models.json with all nine model hashes,
validation choices, protocol and protected training-input fingerprints. Pre-test
deployment choice CNN; primary baseline pixel Random Forest. Downloaded only the
two official test archives and verified publisher checksums and headers.
Evaluated 10,000 official test images (1,000 per class); no model fitting/tuning.
CNN test accuracy 0.890000, macro F1 0.888481.
Pixel RF test accuracy 0.852700, macro F1 0.850921.
Accuracy difference 3.730 percentage points;
paired test-image bootstrap 95% interval [3.140, 4.310] points.
Saved all nine result sets, recalls, confusion matrices, predictions, 27 timing repeats,
complexity details, paired errors, illustrative examples and concise analysis.
CPU benchmark includes preprocessing/HOG, uses common batches and excludes I/O/loading.
Re-executed the full notebook with cached training artifacts; verified metrics and all
frozen hashes. Earlier absence checks now allow official test archives after Step 5 freeze.
The official test set is now used: do not tune on it. Original model weights and split unchanged.
Next: Step 6 reusable upload preprocessing and prediction, using training-only examples
for any new preprocessing decisions; keep final benchmark interpretation unchanged.

## Step 6 completed — 2026-10-20 (Asia/Kolkata)

Added src/predict.py: dataset-free FashionPredictor, explicit preprocessing,
all ten class scores/top predictions, processed preview and CLI JSON output.
Native 28x28 grayscale pixels and scaling exactly match training. Upload adapter
handles RGB/grayscale/alpha, finite normalized floats or uint8, EXIF orientation,
aspect-preserving bilinear resizing, centered dark padding and explicit light inversion.
No automatic segmentation/background guessing, learned adapter or test-driven tuning.
Rejects blank, malformed, corrupt and invalid-range inputs with clear errors.
Saved models/cnn/inference-config.json and training-only notebook examples.
Nine behavioral check groups passed, including score equivalence, unchanged model state
and isolated offline CLI serving without datasets/reports/training modules.
Protected hashes confirm frozen weights, splits and previous evaluation reports unchanged.
No official test images or final scores were used to choose preprocessing.
Next: Step 7 Gradio demo reusing FashionPredictor; do not train or download on startup.

## Step 7 completed — 2026-10-20 (Asia/Kolkata)

Replaced app.py placeholder with a Gradio 6.30.0 demo using one cached CPU
FashionPredictor. Upload/filepath input preserves mode/EXIF; explicit dark/light
background, prediction button, all ten scores, enlarged 28x28 preview and Clear.
Input/background changes reset stale output; blank/missing images show helpful messages.
Exported 11 small training-only PNG samples with provenance into assets/examples/.
Startup never imports dataset/training/evaluation modules, downloads data or trains.
Local server started at http://127.0.0.1:7860, share=False; no public hosting performed.
Seven verification groups passed: callbacks for all samples, one cached model,
HTTP upload/score/preview equivalence for native/light/RGBA/EXIF files, clear and
empty-input handling, and isolated app construction/inference with blocked network
and no datasets/reports/training modules. Browser sample/predict/light/Clear flows verified.
Saved reports/demo verification reports and screenshot; appended executed Step 7 notebook
section. Protected model/split/inference/evaluation artifact hashes unchanged.
Next: Step 8 hosting. Check current platform requirements and prepare a minimal deploy
package; do not include raw datasets, .venv, training reports or extra model artifacts.

## Repository import verified — 2026-10-21 (Asia/Kolkata)

Imported the copied setup scaffold and completed project into Fashion_Classifier_AML.
Grouped the existing files into seven milestone commits, pushing each separately.
These commits record the import; the earlier dated milestones above describe the
original development and experiment runs. No models were retrained for this import.

Validated Python syntax and notebook/JSON structure. The copied Python 3.11
environment passed package imports, CPU tensor/autograd and torchvision checks.
All nine prediction verification groups and all seven local demo verification
groups passed, including HTTP uploads at http://127.0.0.1:7869 and isolated offline
serving. Frozen weights, splits, inference contracts and prior evaluation artifacts
remained unchanged. Training and final test evaluation were not rerun.

The repository README now identifies fashion-classifier/ as the working project
and the root folders as the original setup scaffold. Local environments, datasets,
trained model artifacts and machine-specific lockfiles remain ignored and are
required separately or must be generated for a fresh clone. Public hosting remains
the next project milestone.
