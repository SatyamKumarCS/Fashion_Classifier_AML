# Reference inspection — Step 1

The supplied outline asks for flattened-pixel and HOG traditional baselines versus a small CNN, with fair comparisons and error analysis. Both attachments were inspected; the original notebook was copied byte-for-byte, not executed or edited.

Reference SHA-256: `5ed441f826430f84f6f70f4eb94d1149fb1508025e55db171113485e45513909`

## Split findings
- Cell 10 loads only official training data using torchvision (`train=True`).
- Cell 19 randomly holds out 20% of those 60,000 examples with seed 42, but does not stratify. Its `X_test` is actually a validation set, not the official test set.
- Cells 39/41 evaluate and plot that holdout each epoch. That is appropriate for validation, but naming and final claims must be corrected.
- Cell 46 reads a separate test CSV for a single prediction. It does not provide a complete official test evaluation.
- There is no early stopping, best-validation checkpoint saving, or full baseline/error comparison.

## Missing external files
- Cell 2 requires `/content/sample_data/Architechture_Diagram.png` (spelling exactly as written). It is absent from the supplied attachments and the path is Colab-specific.
- Cell 46 requires `fashion-mnist_test.csv`, also absent.
- Markdown mentions `fashion-mnist_train.csv`; the active loader instead uses torchvision, so that CSV is not required by the current code.
- Fashion-MNIST is not bundled. Its download belongs to Step 2, not this setup.

## Planned correction (not implemented until Step 2)
Use seed 42 to stratify official training indices into 48,000 training and 12,000 validation images. Persist those indices and reuse them for every model. Fit all learned preprocessing on training data only. Select hyperparameters, CNN checkpoints and stopping epochs with validation only. Reserve all 10,000 official test examples for one final comparison after decisions are fixed. Any data-driven preprocessing statistics must also use training only.

The two convolution blocks are a useful reference. Flattening retains pixel locations in fixed vector positions; it removes an explicit spatial layout rather than deleting all location information. A CNN adds a useful local/shared-filter inductive bias.
