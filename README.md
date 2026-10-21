# Automatic Clothing Category Recognition

Inventory Image Tagging via Handcrafted Features and a Small CNN.

Fashion-MNIST flattened-pixel and HOG baselines compared with a small CNN, plus a local
Gradio demo. The committed CNN scores **89.00%** on the official test set, against 85.27% for
the best classical baseline (pixel Random Forest).

**Status: Step 7 complete.** Training, frozen-model evaluation, reusable prediction and
a tested local Gradio demo are implemented. The notebook is executed. Hosting remains.

```
.
├── src/                 data, baselines, CNN, training, evaluation, prediction, v2 experiment
├── scripts/             setup, reproduction pipeline, verification scripts
├── tests/               pytest suite
├── app.py               Gradio demo
├── notebooks/           executed workflow notebook and the unmodified reference lab
├── assets/examples/     demo sample images
├── reports/             committed results, figures, integrity manifests, milestone history
├── data/  models/       generated datasets and weights (gitignored except placeholders)
├── runs/                generated pipeline and v2 experiment outputs (gitignored)
├── archive/step1-setup/ original Step 1 scaffold, kept for reference only
└── pyproject.toml  uv.lock  Dockerfile  compose.yaml
```

## Setup

Dependencies are declared in `pyproject.toml` and pinned in the committed `uv.lock`
(Python 3.11). Both setups install exactly that lock. Run every command from the
repository root.

**Docker** (recommended; only Docker Desktop or Docker Engine with compose is needed):

```bash
docker compose build                 # image: Python 3.11, CPU-only PyTorch, locked dependencies
docker compose run --rm pipeline     # fresh clone: download data, train, evaluate, install models/
docker compose up -d app             # Gradio demo at http://127.0.0.1:7860
docker compose run --rm verify       # artifact hashes, prediction contract, HTTP checks against app
docker compose up notebook           # JupyterLab at http://127.0.0.1:8888 (login URL: docker compose logs notebook)
```

**Native** with [uv](https://docs.astral.sh/uv/). uv downloads Python 3.11 if it's missing:

```bash
bash scripts/setup_mac.sh            # uv sync --locked, environment check, Jupyter kernel
uv run python app.py                 # or: source .venv/bin/activate, then plain python
uv run pytest                        # test suite
```

`pipeline` runs `scripts/reproduce.py`. It downloads the checksum-verified archives and runs
baselines, CNN training and frozen-model test evaluation in a clean `runs/reproduce/`
directory, so the committed reports stay protected. It then installs the trained models into
`models/`, but never replaces different models already there. It also copies the training
archives and split files into `data/` for the verification scripts; the test set is never
copied. Completed stages are skipped on rerun. If a stage fails, delete `runs/reproduce/`
and rerun. Containers have no MPS/GPU, so everything runs on CPU: about 15–20 minutes,
mostly the linear-SVM baselines. Results can differ slightly from the committed MPS run;
see [Model provenance](#model-provenance). `app` mounts `models/` read-only and is limited
to 1 GB and 2 CPUs. `notebook` mounts the whole project read-write, so running its cells can
rewrite reports. Containers run as an unprivileged user.

Run other commands in the same image, for example
`docker compose run --rm app python scripts/check_environment.py`. After changing code,
rebuild the image (`docker compose build`), because the code is copied into the image.
Dependency changes go through uv: edit `pyproject.toml`, run `uv lock`, and commit both files.
Platforms that need a `requirements.txt` can generate one with `uv export`.

In JupyterLab, select **Python (AML Fashion)**. Run All now reuses completed baseline, CNN and test results;
it does not retrain models. Training-only sections never load the official test examples.
Earlier pre-test absence checks permit test archives once the final freeze manifest exists.
The original reference notebook is preserved under `notebooks/reference/`; it is a learning
reference with different validation handling, not the project's final evaluation pipeline.

## Data and fair comparison

Official Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
Archives are verified against the publisher's MD5 checksums and IDX headers.

- Official training: 60,000 grayscale 28x28 images, ten balanced classes.
- Shared seed-42 stratified split: 48,000 training and 12,000 validation examples.
- Actual training budget: the same saved 12,000-example stratified subset for **all nine models** (1,200 per class).
- The remaining 36,000 training examples were unused in this budgeted experiment.
- Official test: 10,000 images, first loaded in Step 5 after model choices were frozen.
- Pixel preprocessing: float32 / 255, original orientation and dark background.
- HOG: 9 bins, 7x7 pixels per cell, 2x2 cells per block, L2-Hys; 324 descriptors.
- LR/SVM StandardScaler is fitted only on the saved baseline training subset and saved inside each pipeline.

The eight classical combinations use pixels/HOG with Logistic Regression, kNN, Random Forest
and linear SVM. Two predeclared settings per combination produced 16 fits; each combination
was selected using validation macro F1, then accuracy. All candidates converged.

The two-block CNN has 429,450 trainable parameters, 32/64 convolution channels, BatchNorm,
pooling, hidden sizes 128/64 and dropout 0.4. It uses Adam, learning rate 0.001, weight decay
0.0001, batch 128, seed 42, at most 25 epochs and five-epoch validation-loss patience.
Training used MPS. It stopped at epoch 14 and restored the best-loss epoch 9. The checkpoint
was verified on CPU. No augmentation was applied. Single-seed results do not establish
variation across training seeds; training/search budgets and selection criteria differ.

## Final evaluation

Before official test access, `reports/final_evaluation/frozen-models.json` recorded model
hashes, validation choices, training-input fingerprints and timing settings. The pre-test
CNN deployment choice and primary pixel Random Forest baseline remain fixed.

Final results cover accuracy, macro F1, per-class recall, confusion matrices, model size,
appropriate complexity measures, CPU timing, paired errors and illustrative mistakes.
Timing uses identical resident image batches (128), two library threads, one warmup batch,
1,024 images and three repeats. It includes scaling/HOG/fitted preprocessing and prediction,
excludes file I/O/model loading/probability generation, and is batch throughput rather than
single-upload latency. Saved sizes reflect different formats; neural parameters, forest
nodes, linear coefficients and stored kNN examples are not equivalent quantities.

Do not tune on these test results and report the same test set as a new untouched evaluation.

## Main files

| Location | Purpose |
|---|---|
| `notebooks/exploration.ipynb` | Executed learning workflow through Step 5 |
| `notebooks/reference/` | Unmodified lab notebook and outline image |
| `src/data.py` | Training-only loader, shared split, pixel scaling |
| `src/baselines.py` | Pixel/HOG baselines and validation selection |
| `src/model.py` | Small CNN architecture |
| `src/train.py` | Validation-based training, early stopping and portable checkpoint |
| `src/evaluate.py` | Frozen-model official-test evaluation, CPU benchmark and analysis |
| `src/predict.py` | Reusable preprocessing, processed preview and dataset-free CNN prediction |
| `models/cnn/inference-config.json` | Upload preprocessing contract and checkpoint fingerprint |
| `scripts/verify_prediction.py` | Behavioral and isolated offline-serving checks |
| `reports/inference/` | Verification, example image/preview/prediction and integrity checks |
| `data/raw/FashionMNIST/` | Verified original training/test archives |
| `data/processed/` | Saved split, baseline training subset and preprocessing contract |
| `models/baselines/` | Eight selected fitted models and learned scaler pipelines |
| `models/cnn/best.pt` | Best CPU-portable state dict, buffers and metadata |
| `models/cnn/config.json` | Readable architecture and training settings |
| `reports/baselines/` | All candidates, selected validation results, predictions and recalls |
| `reports/cnn/` | History, best-checkpoint validation results and predictions |
| `reports/final_evaluation/` | Frozen manifest, all test metrics, timings, predictions and analysis |
| `reports/figures/` | Training samples, learning curves, confusion matrices and error examples |
| `reports/progress.md` | Dated milestone handoff notes |
| `reports/step*-execution.log` | Actual local execution logs |
| `reports/step*-backup/` | Preserved working files before each milestone |
| `scripts/setup_mac.sh`, `scripts/check_environment.py` | Native uv setup and environment verification |
| `app.py` | Local upload-and-predict Gradio demo using the saved CNN |
| `assets/examples/` | Eleven pre-exported training-only samples and their provenance |
| `scripts/verify_demo.py` | Callback, HTTP-upload and isolated-startup tests |
| `reports/demo/` | Verification reports, browser screenshot and protected artifact fingerprints |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependency declaration and the exact locked environment |
| `Dockerfile`, `compose.yaml`, `.dockerignore` | Container image and the pipeline/app/verify/notebook services |
| `scripts/reproduce.py` | Full pipeline in `runs/reproduce/`, then model installation with provenance |
| `scripts/verify_artifacts.py` | Checks every hash-protected file and identifies the installed models |
| `models/PROVENANCE.json` | Origin, hashes and test accuracy of reproduced models (absent for original weights) |
| `src/train_v2.py` | v2 ablation experiment: training, validation-only selection, one test evaluation |
| `tests/` | pytest suite: integrity, prediction contract, v2 rule and guards, install safety |
| `runs/` | Generated pipeline and v2 outputs (gitignored) |

`reports/environment-*.txt` from the native setup describes the local machine; the
workspace/Linux reports are historical records from a different environment. Data, model
weights and `runs/` are excluded by `.gitignore`; preserve them separately.

`archive/step1-setup/` holds the original Step 1 scaffold, kept for reference. The project
doesn't use it, and it's excluded from the Docker image.

## Model provenance

The checkpoints behind the committed results (CNN 89.00% test accuracy) are not in this
repository; `reports/final_evaluation/frozen-models.json` records their hashes. A fresh
clone gets its models from `pipeline`, which retrains them on CPU with the same code and a
byte-identical data split. The committed CPU reproduction scored **88.91%** for the CNN and
85.27% for pixel Random Forest. These are reproduced models, not the original checkpoints, so
`reproduce.py` records them in `models/PROVENANCE.json`, together with the matching manifest
`models/reproduced-sha256.json`.

`python scripts/verify_artifacts.py` reports which case applies. Either way, every tracked
report, source file and the reference notebook must match the committed hashes. Installed
models must match either the original frozen hashes or their provenance record.
`verify_demo.py` picks the right manifest automatically (`--manifest` overrides). The
notebook's Step 6 verification cell compares the checkpoint against the original contract,
so it only passes with the original weights.

## v2 experiment (in progress)

`src/train_v2.py` asks whether the 12,000-image budget, the lack of augmentation and the
small CNN limit accuracy. Each of five configurations changes one factor relative to its
comparison. All share Adam (lr 1e-3, weight decay 1e-4), batch 128, early stopping on
validation loss (patience 5, at most 40 epochs) and the seed-42 split. Each runs with
seeds 42, 1 and 2:

| Config | Training images | Augmentation | Network |
|---|---|---|---|
| `v1_12k` | 12,000 (v1 subset) | – | v1 SmallCNN (reference, inside the v2 training loop) |
| `v1_48k` | 48,000 | – | v1 SmallCNN |
| `v1_48k_aug` | 48,000 | flip + ≤2 px shift | v1 SmallCNN |
| `deep_48k` | 48,000 | – | 3×(2 conv) blocks, 585k parameters |
| `deep_48k_aug` | 48,000 | flip + ≤2 px shift | 3×(2 conv) blocks |

The train–validation gap is measured in eval mode, without augmentation, on the same
12,000 v1 training images for every configuration.

```bash
python -m src.train_v2 train --config deep_48k --seed 42   # one run → runs/v2/experiments/
python -m src.train_v2 report                              # validation comparison table
python -m src.train_v2 select                              # freeze the choice (validation only)
python -m src.train_v2 test                                # official test set, exactly once
```

Selection uses validation data only, through a paired two-level bootstrap. Validation
images are resampled jointly for each pair of configurations, and seed runs are resampled
within each configuration. The rule picks the simplest configuration whose accuracy is not
distinguishable from the best, and adopts it only if it clearly beats `v1_12k`. Otherwise
v1 stays and the test set is not touched. The full rule and its revision history are in
`SELECTION_RULE` and `RULE_HISTORY` and are copied into `runs/v2/selection.json`. Training
is refused once a selection exists. The test evaluation writes a marker before opening the
test set, so it can't be repeated. The v1 code, models, reports and 89.00% result are only
read, never modified.

## Reproducibility

With Docker, `docker compose run --rm pipeline` does all of the following in a separate run
directory. For a fresh project copy without completed artifacts in the project itself, run
these commands in order from the project root, inside the environment (`uv run …` or the
`app` image):

```bash
python -m jupyterlab notebooks/exploration.ipynb
```

The notebook loads training data, saves the split, trains baselines/CNN if results are
missing, freezes choices and conducts final evaluation. For separate milestone execution:

```bash
python -m src.baselines
python -m src.train
python -m src.evaluate
```

Existing completed runs are protected against overwriting. `src.baselines` needs Step 2
archives/split; `src.train` also needs its saved baseline subset; `src.evaluate` needs all
selected models. A CNN checkpoint includes model state and metadata, not optimizer state,
so exact training resumption is not implemented.

## Next milestones

Step 8: choose hosting using current requirements, then document and verify deployment.
Fashion-MNIST results apply to benchmark thumbnails, not arbitrary real clothing photos.


## Step 6 — reusable prediction

`FashionPredictor` loads the saved CNN once. It accepts an image path, PIL image or NumPy
array, returns class name/index, ten softmax scores, ranked predictions and a processed
28x28 preview. The inference module imports neither dataset loaders nor training modules.
It defaults to CPU; MPS can be selected explicitly if available.

```python
from src.predict import FashionPredictor
predictor = FashionPredictor()
result = predictor.predict("your-image.png", background="dark", top_k=3)
print(result.label)
print(result.scores)
result.preview.save("processed-preview.png")
```

From the active environment in the project root, try the saved **training** example:

```bash
python -m src.predict reports/inference/example-input.png --preview /tmp/example-preview.png
python scripts/verify_prediction.py
```

The preview goes to `/tmp` because `reports/inference/example-preview.png` is a
hash-protected record.

CLI prediction prints JSON. Use `--background light` for a light-background input.
Native nonblank 28x28 uint8 grayscale on a dark background is pixel-identical to training.
For other uploads, EXIF orientation is respected, RGB is converted to grayscale, alpha is
composited onto the chosen background, aspect ratio is preserved during resize and black
padding centers the 28x28 result. Light background is explicitly inverted, not guessed.
No cropping, segmentation, learned background removal or test-driven tuning is performed.
Float arrays must be finite in [0,1]; uint8 arrays represent [0,255]. Other array dtypes,
invalid shapes, blank inputs, corrupt files and missing images produce clear errors.

The upload contract is stored in `models/cnn/inference-config.json`. This adapter is not
a new accuracy claim on real photographs. Clothing should be isolated, upright and
clearly visible; busy backgrounds and different framing can cause failures. Softmax
scores are not calibrated probabilities of being correct. The preview shows exactly
what reaches the model. Existing benchmark results remain unchanged.

Behavioral checks passed for native pixel/scaling identity and direct-model scores,
all ten class labels, color/alpha conversion, explicit inversion, arrays, EXIF, letterbox,
invalid inputs, unchanged weights/BatchNorm buffers, and isolated CLI prediction with
network blocked and no datasets/reports/training modules present. Serving requires the
checkpoint, `src/__init__.py`, `src/model.py`, `src/predict.py`, and NumPy/Pillow/PyTorch.
The verification script itself uses training archives to check preprocessing equivalence;
it is not needed by the eventual demo at startup.

`app.py` implements the Step 7 demo below. Prediction never trains or downloads data.


## Step 7 — local Gradio demo

From the project root, either `docker compose up -d app`, or natively:

```bash
uv run python app.py
```

Open http://127.0.0.1:7860. Upload an image or choose a training sample, pick the image's
dark/light background and click **Predict category**. The app shows the predicted category,
all ten scores and the exact 28x28 processed preview. **Clear** resets everything. Input
changes clear stale outputs; blank/missing inputs get a helpful message and empty results.
Scores are softmax values, not calibrated confidence, and arbitrary photos may work poorly.

`python app.py --open-browser` also opens the page. Keep the Terminal running and press
Control+C there to stop your server. If 7860 is occupied, use `python app.py --port 7861`.
Default binding is local (127.0.0.1) with `share=False`; this step does not publish a site.

The app loads one cached CPU predictor, uses a bounded queue, and preserves uploaded
color mode/EXIF until `src.predict` handles them. Analytics is disabled and the theme uses
system fonts. Display scaling enlarges tiny thumbnails without altering model pixels.
Startup imports no training/dataset/evaluation module and never downloads data or trains.
Example PNGs are exported once from the saved training subset; provenance is in
`assets/examples/manifest.json`. They are illustrative samples, not a new accuracy claim.

With the server running (in Docker, `docker compose run --rm verify` does this too):

```bash
uv run python scripts/verify_demo.py
```

For another port, add `--url http://127.0.0.1:7861`. This checks callback equivalence,
one cached model, actual HTTP uploads (PNG, light background, RGBA and EXIF JPEG), score
and preview equality, missing/blank input and Clear, isolated startup with blocked network
and no data/reports/training modules, and unchanged prior artifacts. Browser testing also
verified samples, prediction, background choice and Clear. Reports and a screenshot are
saved under `reports/demo/`; original pre-Step-7 files are in `reports/step6-backup/`.

A deployable demo needs `app.py`, `src/__init__.py`, `src/model.py`, `src/predict.py`,
`models/cnn/best.pt`, optional `assets/examples/`, and compatible Gradio/NumPy/Pillow/PyTorch
libraries. It needs no raw datasets or training reports. Public hosting and a deployment
package belong to Step 8; the existing benchmark scores remain unchanged.
