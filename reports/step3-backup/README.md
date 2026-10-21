# Automatic Clothing Category Recognition

Status: Step 3 complete. Eight budgeted pixel/HOG baselines trained and validated. Official test data reserved.

Compare Logistic Regression, kNN, Random Forest and a suitable SVM on flattened pixels and HOG features with a small CNN. Later milestones add fair evaluation, error analysis, reusable prediction, a Gradio demo and hosting.

## Mac setup

Extract this folder inside `~/Desktop/AML_Project`. Keep any existing files. If a `fashion-classifier` folder already exists, choose another folder name rather than overwriting it.

Recommended interpreter: Python 3.11. Check `python3.11 --version` and `uname -m`. If Python 3.11 is missing and you have Homebrew, run `brew install python@3.11`; otherwise install Python 3.11 from python.org. Apple Silicon can use MPS when available; Intel Mac uses CPU and a legacy torch/torchvision pair. Native Intel support is frozen in that older pair, so keep NumPy below 2. Use native Python matching your Mac architecture when possible.

```bash
cd "$HOME/Desktop/AML_Project/fashion-classifier"
bash scripts/setup_mac.sh
source .venv/bin/activate
python -m jupyterlab notebooks/exploration.ipynb
```

The setup script creates `.venv` (an isolated project environment), installs dependencies with that environment's Python, runs `pip check`, verifies imports and synthetic tensor operations, records resolved versions in `requirements-lock-local.txt`, and registers the notebook kernel `Python (AML Fashion)`. It stops if a check fails. Do not copy a Linux `.venv` to a Mac; recreate it locally.

Select `Python (AML Fashion)` in Jupyter. Use `notebooks/exploration.ipynb` for setup and Step 2 data inspection. The preserved reference notebook includes dataset downloads and training, so do not Run All on it during setup.

## Files

- `notebooks/exploration.ipynb`: setup checks, training-data inspection, and saved stratified split.
- `notebooks/reference/`: unchanged original lab and outline image.
- `scripts/check_environment.py`: imports, CPU matrix multiplication, gradients, torchvision binary compatibility and available accelerator check; no network/data calls.
- `scripts/setup_mac.sh`: repeatable setup and verification.
- `src/data.py`: training-only loader, checksum verification, persistent split, and pixel scaling. `src/baselines.py`: budgeted pixel/HOG model training and selection. Other modules remain placeholders.
- `app.py`: placeholder until the demo milestone.
- `data/raw`, `data/processed`, `models`, `reports/figures`: empty output locations.
- `reports/reference-review.md`: detailed inspection and future split correction.
- `reports/verification-workspace.txt`: actual workspace verification outcome, distinct from your Mac.

## Evaluation contract

Shared split: official training data is divided into stratified train/validation indices with seed 42. Use the same split for every approach. Keep official test data out of inspection, tuning and training curves until final evaluation. Step 2 implements this split and training-only data pipeline in `src/data.py`.

Final metrics: accuracy, macro F1, per-class recall, confusion matrices and consistent inference timing. CNN trainable parameters will be reported. For classical models, report meaningful size/complexity measures rather than pretending trees, stored kNN samples and support vectors have directly comparable neural parameter counts.

## Step 1 completion on your Mac

The check must end with `ALL CHECKS PASSED`. Preserve `reports/environment-local.txt` and `requirements-lock-local.txt`. If it fails, send the error output, `python3.11 --version`, and `uname -m` before proceeding. See `reports/reference-review.md` for the reference's missing files; these are not needed for setup.

## Setup documentation sources

- PyTorch installation: https://pytorch.org/get-started/locally/
- Official previous-version pairs: https://pytorch.org/get-started/previous-versions/
- Intel Mac build status: https://discuss.pytorch.org/t/why-no-macosx-x86-64-build-after-torch-2-2-2-cp39-none-macosx-10-9-x86-64-whl/204546/2
- Gradio environment setup: https://www.gradio.app/guides/installing-gradio-in-a-virtual-environment


## Step 2 — training data and saved split

Run `notebooks/exploration.ipynb` top to bottom using **Python (AML Fashion)**.
The first Step 2 run downloads only the two official training archives (about 26 MB)
from Zalando's GitHub repository and verifies the publisher's MD5 checksums.
Later runs reuse the verified local archives. No official test file is downloaded or loaded.

- `src/data.py`: reusable training loader, split validation, and float32 pixel scaling.
- `data/raw/FashionMNIST/`: the two original compressed training archives.
- `data/processed/train_validation_split.npz`: exact original-row indices; seed 42, stratified 48,000/12,000 split.
- `data/processed/preprocessing.json`: labels, original orientation, pixel scaling, and split settings.
- `reports/class_counts.csv`: 6,000 examples per category, divided into 4,800 training and 1,200 validation.
- `reports/figures/training_examples.png`: three examples per category from training indices only.
- `reports/step2-summary.json`: measured checks and versions.
- `reports/step1-backup/`: copies of the working files before Step 2.

Pixel scaling uses `uint8 -> float32 / 255`, preserving 28x28 shape and orientation.
Future learned preprocessing must be fitted using training indices only. Keep the saved
split for every baseline and CNN. A mismatched existing split or config raises an error
rather than being overwritten. No model training was performed during Step 2; Step 3 baseline results are documented below.

Dataset source: https://github.com/zalandoresearch/fashion-mnist

Step 3 adds flattened-pixel and HOG baselines below. The 10,000 official test examples remain
reserved for final evaluation after model choices are frozen.


## Step 3 — budgeted traditional baselines

The executed exploration notebook now includes baseline explanations, a results table,
a comparison chart, and saved-model reload checks. `python -m src.baselines` runs training
from the project root; it refuses to overwrite a completed run. Notebook Run All reuses
completed baseline results instead of training again. Models use the existing seed-42
split and a shared 12,000-image stratified subset of its training portion, saved in
`data/processed/baseline_train_indices.npy`. All 12,000 validation rows are scored.

Both flattened pixels (784 values) and HOG (324 values) are compared with Logistic
Regression, kNN, Random Forest, and linear SVM. Two settings per combination give 16
candidate fits. Selection uses validation macro F1, breaking ties with accuracy;
non-converged candidates are excluded. Training-only StandardScaler pipelines are saved
with Logistic Regression and SVM; kNN and forests use descriptors directly.

- `reports/baselines/candidates.csv`: all settings, validation scores, convergence warnings and timings.
- `reports/baselines/validation_results.csv`: eight selected configurations, sorted by macro F1.
- `reports/baselines/validation_predictions.npz`: validation indices, labels and selected predictions.
- `reports/baselines/*_recall.csv`: per-class validation recall.
- `reports/baselines/summary.json`: feature settings, budget, checksums, selection rule and versions.
- `models/baselines/*.joblib`: eight selected locally fitted models, including any learned scaler.
- `reports/figures/baseline_validation_macro_f1.png`: validation comparison.
- `reports/step3-execution.log`: local training log.
- `reports/step2-backup/`: preserved pre-Step-3 working files.

These are validation selection scores on budgeted training data, not official test scores.
Timing is preliminary local CPU timing, excludes descriptor extraction during prediction,
and must not be used as the final end-to-end latency comparison. Saved compressed size
is not a neural-network-equivalent parameter count. A final comparison must document or
align training budgets if the CNN uses more training examples. No official test file
has been downloaded or loaded. Next milestone: Step 4, CNN training with early stopping.
