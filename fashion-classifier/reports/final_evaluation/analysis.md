# Final official-test evaluation

All nine model configurations and the primary CNN/Random Forest comparison were frozen before official test loading. No fitting or hyperparameter search occurred in this milestone.

CNN: accuracy **89.00%**, macro F1 **0.8885**. Pixel Random Forest: accuracy **85.27%**, macro F1 **0.8509**.

Accuracy difference: **3.73 percentage points**. Paired test-image bootstrap 95% interval: **3.14 to 4.31 points** (2,000 resamples, seed 42). This interval does not include training-seed variability.

CNN correct/RF wrong: 617; RF correct/CNN wrong: 244; both wrong: 856; both correct: 8283.

## Per-class recall

| Category | Pixel Random Forest | CNN |
|---|---:|---:|
| T-shirt/top | 82.2% | 86.5% |
| Trouser | 94.2% | 96.8% |
| Pullover | 78.6% | 82.5% |
| Dress | 88.9% | 89.6% |
| Coat | 78.8% | 88.5% |
| Sandal | 93.9% | 94.9% |
| Shirt | 52.9% | 59.9% |
| Sneaker | 92.2% | 96.9% |
| Bag | 96.3% | 97.9% |
| Ankle boot | 94.7% | 96.5% |

## Frequent upper-body confusions

pixels_random_forest: Shirt → T-shirt/top: 170; Shirt → Pullover: 129; Pullover → Coat: 124; Coat → Pullover: 124; Shirt → Coat: 118.
cnn: Shirt → T-shirt/top: 148; Shirt → Coat: 127; Pullover → Coat: 111; Shirt → Pullover: 83; T-shirt/top → Shirt: 74.

## Interpretation and limits

The CNN improves recall in all ten categories in this run. Coat recall gains 9.7 percentage points (78.8% to 88.5%). Shirts remain the weakest category: recall rises from 52.9% to 59.9%, leaving 401 of 1,000 shirts misclassified. Shirt-to-coat errors increase from 118 to 127 despite the overall improvement; the CNN does not resolve every specific confusion.

The paired outcomes show where learned spatial features recover baseline errors and where the CNN still fails. Examples in the figure are fixed-seed illustrations prioritizing shirts, T-shirts/tops, pullovers and coats; they are not a representative sample of all errors.

CPU timing includes descriptor extraction and fitted preprocessing, with identical image batches and three warmed repetitions. Raw repeats are saved. It excludes decoding, file access and model loading. HOG must be computed for new images and its cost is included.

The complexity report uses weights for linear models, tree nodes for forests, stored examples/features for kNN, and trainable parameters for the CNN. These quantities are not directly equivalent. File sizes reflect the actual saved serialization format.

- Single trained seed and 12000-image training subset for every model.
- Different optimization/search budgets and checkpoint selection criteria.
- Timing uses resident arrays, warm models, batch size 128 and CPU only; not upload latency or single-image latency.
- Serialized sizes use different formats; tree nodes, stored samples and neural parameters are not equivalent.
- Fashion-MNIST grayscale thumbnails do not establish performance on real product photos.
- Official test results are final reporting; further changes require new validation-based decisions, not test-driven tuning.

The deployment choice remains the CNN selected using validation before test evaluation. Next milestone: reusable upload-image preprocessing and prediction, matching the saved input contract.
