# Step 1 setup scaffold (archived)

This is the original Step 1 environment scaffold, kept for reference. It isn't used by the
project in `fashion-classifier/`, which now uses `pyproject.toml` and `uv.lock`.

Moved here from the repository root:

- `notebooks/00_setup.ipynb`: the Step 1 setup notebook. It finds `scripts/check_environment.py` in its parent folders, so it still runs from here.
- `scripts/check_environment.py`: the Step 1 dependency check. The project has a newer version.
- `requirements.txt`: the unpinned Step 1 package list. `pyproject.toml` and `uv.lock` replace it.

Removed after verification:

- `notebooks/reference/LAB_12_CNN_Fashion_mnist.ipynb`: byte-identical to the hash-protected copy in `fashion-classifier/notebooks/reference/`.
- `data/`, `models/`, `reports/figures/`: empty placeholder folders.
- `src/__init__.py`: a one-line placeholder.

The original notes follow unchanged; their paths refer to the old layout.

---

# AML Project: Automatic Clothing Category Recognition

Compare flattened-pixel and HOG classifiers with a small CNN on Fashion-MNIST.

## Step 1: setup on macOS

Use Python 3.11 or 3.12. Check with `python3 --version`; if it differs, use a
Python 3.11/3.12 executable for the venv command. The following commands assume
that this archive's contents have been extracted into `~/Desktop/AML_Project`.

```bash
cd "$HOME/Desktop/AML_Project"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

Only continue if the check ends with `STEP 1 CHECKS PASSED`.
Then record the versions that resolved on your machine and start Jupyter:

```bash
python -m pip freeze > requirements-lock.txt
python -m jupyter lab
```

Open `notebooks/00_setup.ipynb` and run all cells. Use the kernel belonging to
this project's `.venv`. CPU is a valid fallback if CUDA or Apple MPS is absent.
The version lock is specific to the machine where it was generated; deployment
will get its own validated requirements later.

## Folder purposes

- `notebooks/00_setup.ipynb`: setup checks and the next-step checklist.
- `notebooks/reference/LAB_12_CNN_Fashion_mnist.ipynb`: untouched uploaded lab.
- `src/`: reusable data, feature, model, and evaluation code added in later steps.
- `scripts/`: commands for checking the environment and, later, training.
- `data/`: downloaded dataset; excluded from Git.
- `models/`: saved model weights; excluded from Git by default.
- `reports/figures/`: generated charts and confusion matrices.

## Reference notebook status

The uploaded lab is included unchanged as a reference, not a runnable final
project. It refers to an external architecture PNG and test CSV. Its 80/20 split
comes from the official training set; we will call the held-out portion
validation and load the official test set separately. We will also add ML/HOG
baselines, early stopping, saved weights, metrics, and inference timing.

## Roadmap

1. Environment and folder setup (this package).
2. Dataset inspection and shared stratified train/validation split; preserve
   the official test set for final evaluation.
3. Pixel and HOG features with Logistic Regression, kNN, Random Forest, and SVM.
4. CNN training, validation, early stopping, and checkpoint saving.
5. Final test metrics, timing, and error analysis.
6. Shared image preprocessing and prediction function.
7. Gradio interface.
8. Hosting and deployment documentation.

A Fashion-MNIST model is a benchmark prototype. Arbitrary product photographs
may differ substantially from its 28x28 grayscale training images.
