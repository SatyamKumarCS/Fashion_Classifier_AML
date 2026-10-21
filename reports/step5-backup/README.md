# Automatic Clothing Category Recognition

Inventory Image Tagging via Handcrafted Features and a Small CNN.

**Status: Step 5 complete.** Training, validation selection and frozen-model official-test
evaluation are complete. The notebook is executed and reusable modules are implemented.
A simple upload-and-predict interface and hosting are later milestones.

## Open the project

The working Mac environment already exists. Do not recreate it unnecessarily.

```bash
cd "$HOME/Desktop/AML_Project/fashion-classifier"
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
| `app.py` | Placeholder for the later Gradio demo |

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

Step 6: reusable image preprocessing and prediction matching the saved CNN input contract.
Step 7: Gradio upload-and-predict demo that never downloads training data or trains on startup.
Step 8: choose hosting using current requirements, then document and verify deployment.
Fashion-MNIST results apply to benchmark thumbnails, not arbitrary real clothing photos.
