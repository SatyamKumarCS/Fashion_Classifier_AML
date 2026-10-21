# Automatic Clothing Category Recognition

Inventory Image Tagging via Handcrafted Features and a Small CNN.

**Status: Step 7 complete.** Training, frozen-model evaluation, reusable prediction and
a tested local Gradio demo are implemented. The notebook is executed. Hosting remains.

## Open the project

For this copied workspace, reuse the existing Mac environment if its Python executable
works. Fresh clones should run `bash scripts/setup_mac.sh` from `fashion-classifier/`
to create an environment, then generate the required data and models as described below.

```bash
cd fashion-classifier  # from the repository root
source .venv/bin/activate
python -m jupyterlab notebooks/exploration.ipynb
```

Select **Python (AML Fashion)**. Run All now reuses completed baseline, CNN and test results;
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
| `scripts/` | Original Mac setup and environment verification |
| `app.py` | Local upload-and-predict Gradio demo using the saved CNN |
| `assets/examples/` | Eleven pre-exported training-only samples and their provenance |
| `scripts/verify_demo.py` | Callback, HTTP-upload and isolated-startup tests |
| `reports/demo/` | Verification reports, browser screenshot and protected artifact fingerprints |

The local lockfile and environment report describe this Mac; workspace/Linux reports are
historical records from a different environment. Large data/model artifacts and local
lockfiles/reports are excluded by the existing `.gitignore`; preserve them separately.

## Reproducibility

For a fresh project copy without completed artifacts, run these commands in order from
the project root after using the original setup script:

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
python -m src.predict reports/inference/example-input.png --preview reports/inference/example-preview.png
python scripts/verify_prediction.py
```

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

From the existing environment in the project root:

```bash
source .venv/bin/activate
python app.py
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

With the server running:

```bash
python scripts/verify_demo.py
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
