# Automatic Clothing Category Recognition

Inventory image tagging with handcrafted features and a small CNN.

This project classifies Fashion-MNIST clothing images into ten categories. It compares
classical models (on flattened pixels and HOG features) with a small convolutional network
and serves the best model through a local Gradio demo. On the official test set, the CNN
reaches **89.00%** accuracy, against **85.27%** for the best classical baseline (pixel
Random Forest).

## Team

| Name | Enroll |
|---|---|
| Satyam Kumar | 2401010428 |
| Krishna Verma | TBD |
| Akshit Vats | TBD |
| Chinmay Soni | 2401010134 |
| Lakshya Bapna | TBD |

## Results

All models are trained on the same 12,000-image subset and evaluated once on the 10,000
official test images.

| Model | Test accuracy | Test macro F1 | Validation accuracy |
|---|---|---|---|
| **CNN** | **89.00%** | **88.85%** | 89.93% |
| Pixels + Random Forest | 85.27% | 85.09% | 86.46% |
| HOG + Logistic Regression | 83.27% | 83.22% | 83.94% |
| HOG + Linear SVM | 82.56% | 82.40% | 84.03% |
| Pixels + kNN | 82.41% | 82.40% | 83.47% |
| Pixels + Logistic Regression | 82.27% | 82.22% | 83.75% |
| HOG + Random Forest | 81.97% | 81.94% | 82.84% |
| HOG + kNN | 81.09% | 81.15% | 82.24% |
| Pixels + Linear SVM | 80.03% | 79.84% | 81.93% |

Per-class recall, confusion matrices, CPU timings, model sizes and error analysis are in
`reports/final_evaluation/`. Figures are in `reports/figures/`.

## Project layout

```
.
├── src/                 data, baselines, CNN, training, evaluation, prediction, v2 experiment
├── scripts/             setup, reproduction pipeline, verification scripts
├── tests/               pytest suite
├── app.py               Gradio demo
├── notebooks/           executed workflow notebook and the unmodified reference lab
├── assets/examples/     demo sample images
├── reports/             committed results, figures, integrity manifests, milestone history
├── data/  models/       generated datasets and weights (gitignored)
├── runs/                generated pipeline and v2 experiment outputs (gitignored)
├── archive/step1-setup/ original Step 1 scaffold, kept for reference only
└── pyproject.toml  uv.lock  Dockerfile  compose.yaml
```

## Setup

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock` (Python 3.11). Both
setups install exactly that lock. Run every command from the repository root.

**Docker** (recommended; needs only Docker with compose):

```bash
docker compose build                 # Python 3.11, CPU-only PyTorch, locked dependencies
docker compose run --rm pipeline     # download data, train, evaluate, install models/
docker compose up -d app             # Gradio demo at http://127.0.0.1:7860
docker compose run --rm verify       # artifact hashes, prediction contract, HTTP checks
docker compose up notebook           # JupyterLab at http://127.0.0.1:8888
```

`pipeline` runs the whole workflow on CPU in `runs/reproduce/`, so the committed reports are
never overwritten. It takes about 15–20 minutes, mostly for the linear-SVM baselines.
Rebuild the image after changing code, because the code is copied into it.

**Native** with [uv](https://docs.astral.sh/uv/) (uv downloads Python 3.11 if needed):

```bash
bash scripts/setup_mac.sh            # uv sync --locked, environment check, Jupyter kernel
uv run python app.py                 # Gradio demo
uv run pytest                        # test suite
```

To change dependencies, edit `pyproject.toml`, run `uv lock`, and commit both files.

## Data and method

- **Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), 28×28
  grayscale images in ten balanced classes. Archives are checked against the publisher's MD5
  checksums.
- **Split:** a stratified split with seed 42 divides the 60,000 training images into 48,000
  training and 12,000 validation images.
- **Training budget:** every model uses the same saved 12,000-image subset (1,200 per class).
- **Test set:** the official 10,000 test images were loaded only after all model choices
  were frozen in `reports/final_evaluation/frozen-models.json`.
- **Features:**
  - Pixels: scaled to [0, 1].
  - HOG: 9 bins, 7×7 cells, 2×2 blocks, giving 324 descriptors.
- **Baselines:**
  - Logistic Regression, kNN, Random Forest and linear SVM, each on pixels and on HOG.
  - Two predeclared settings per combination.
  - Each combination is selected by validation macro F1.
- **CNN:**
  - Architecture: two convolution blocks (32/64 channels) with BatchNorm and pooling, dense
    layers of 128 and 64, and dropout 0.4. It has 429,450 parameters.
  - Training: Adam (learning rate 1e-3, weight decay 1e-4) with batch size 128. Early stopping
    watches validation loss with patience 5; the best epoch (9 of 14) was kept.

## Prediction and demo

`FashionPredictor` loads the saved CNN once. It accepts an image path, a PIL image or a NumPy
array, and returns:

- the predicted label;
- all ten softmax scores;
- the ranked predictions;
- the exact 28×28 preview passed to the model.

```python
from src.predict import FashionPredictor

result = FashionPredictor().predict("your-image.png", background="dark", top_k=3)
print(result.label, result.scores)
```

From the command line: `python -m src.predict image.png` (add `--background light` for
light backgrounds).

**Demo:** run `uv run python app.py` (or `docker compose up -d app`) and open
http://127.0.0.1:7860.

1. Upload an image or pick a sample.
2. Choose whether the image has a dark or light background.
3. Click **Predict category**.

The app binds to localhost only and never trains or downloads data.

**Limitations:**

- The model was trained on benchmark thumbnails. Real clothing photos work best when the
  item is isolated, upright and on a plain background.
- Scores are softmax values, not calibrated confidence.

## Verification

```bash
python scripts/verify_artifacts.py   # every hash-protected report, source and model file
python scripts/verify_prediction.py  # preprocessing and prediction contract
python scripts/verify_demo.py        # demo callbacks and HTTP uploads (server must be running)
uv run pytest                        # integrity, prediction, v2 safeguards, install safety
```

## Model provenance

The original 89.00% checkpoints aren't in the repository; their hashes are recorded in
`reports/final_evaluation/frozen-models.json`. A fresh clone gets its models from `pipeline`,
which retrains them on CPU using the same code and the same data split.

The CPU reproduction scored **88.91%** for the CNN and 85.27% for pixel Random Forest. The
original run used MPS, so small differences are expected. Reproduced models are recorded in
`models/PROVENANCE.json`, and `verify_artifacts.py` reports which set is installed.

## v2 experiment

`src/train_v2.py` is a controlled ablation that tests whether accuracy improves with:

- all 48,000 training images;
- mild augmentation (horizontal flip plus a shift of up to 2 px);
- a deeper CNN.

Each configuration changes one factor and runs with seeds 42, 1 and 2.

| Config | Training images | Augmentation | Network |
|---|---|---|---|
| `v1_12k` | 12,000 | – | v1 CNN (reference) |
| `v1_48k` | 48,000 | – | v1 CNN |
| `v1_48k_aug` | 48,000 | flip + shift | v1 CNN |
| `deep_48k` | 48,000 | – | deeper CNN (585k parameters) |
| `deep_48k_aug` | 48,000 | flip + shift | deeper CNN |

```bash
python -m src.train_v2 train --config deep_48k --seed 42   # one run → runs/v2/experiments/
python -m src.train_v2 report                              # validation comparison table
python -m src.train_v2 select                              # freeze the choice (validation only)
python -m src.train_v2 test                                # official test set, exactly once
```

Selection uses validation data only, through a paired bootstrap. It picks the simplest
configuration that isn't distinguishable from the best, and adopts it only if it clearly
beats `v1_12k`. The test set can be evaluated only once. The v1 code, models and the 89.00%
result are never modified.

## Next steps

Step 8: choose a hosting platform, then deploy and verify the demo.
