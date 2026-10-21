# Automatic Clothing Category Recognition

Status: Step 1 starter. No data downloaded and no training performed.

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

Select `Python (AML Fashion)` in Jupyter. Run only `notebooks/exploration.ipynb` now. The preserved reference notebook includes dataset downloads and training, so do not Run All on it during setup.

## Files

- `notebooks/exploration.ipynb`: setup-only notebook.
- `notebooks/reference/`: unchanged original lab and outline image.
- `scripts/check_environment.py`: imports, CPU matrix multiplication, gradients, torchvision binary compatibility and available accelerator check; no network/data calls.
- `scripts/setup_mac.sh`: repeatable setup and verification.
- `src/{data,baselines,model,train,evaluate}.py`: documented placeholders for future milestones.
- `app.py`: placeholder until the demo milestone.
- `data/raw`, `data/processed`, `models`, `reports/figures`: empty output locations.
- `reports/reference-review.md`: detailed inspection and future split correction.
- `reports/verification-workspace.txt`: actual workspace verification outcome, distinct from your Mac.

## Evaluation contract

Next milestone: split only official training data into stratified train/validation indices with seed 42. Use the same split for every approach. Keep official test data out of inspection, tuning and training curves until final evaluation. No split or dataset pipeline has been implemented in Step 1.

Final metrics: accuracy, macro F1, per-class recall, confusion matrices and consistent inference timing. CNN trainable parameters will be reported. For classical models, report meaningful size/complexity measures rather than pretending trees, stored kNN samples and support vectors have directly comparable neural parameter counts.

## Step 1 completion on your Mac

The check must end with `ALL CHECKS PASSED`. Preserve `reports/environment-local.txt` and `requirements-lock-local.txt`. If it fails, send the error output, `python3.11 --version`, and `uname -m` before proceeding. See `reports/reference-review.md` for the reference's missing files; these are not needed for setup.

## Setup documentation sources

- PyTorch installation: https://pytorch.org/get-started/locally/
- Official previous-version pairs: https://pytorch.org/get-started/previous-versions/
- Intel Mac build status: https://discuss.pytorch.org/t/why-no-macosx-x86-64-build-after-torch-2-2-2-cp39-none-macosx-10-9-x86-64-whl/204546/2
- Gradio environment setup: https://www.gradio.app/guides/installing-gradio-in-a-virtual-environment
