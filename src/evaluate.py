"""Frozen-model official-test evaluation, CPU timing, and error analysis.

Run from the project root: python -m src.evaluate
Models are never fitted here. Model hashes and choices are recorded before test loading.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import platform
import struct
import time
import urllib.request

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import skimage
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from threadpoolctl import threadpool_limits

from .data import CLASS_NAMES, SOURCE, scale_pixels
from .baselines import features, HOG_CONFIG
from .model import SmallCNN, count_parameters

TEST_FILES = {'t10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
              't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310'}
PROTOCOL = {'batch_size': 128, 'timing_images': 1024, 'timing_repeats': 3,
            'warmup_batches': 1, 'timing_device': 'cpu', 'library_threads': 2,
            'timing_input': 'first 1024 official test images in original order, held in memory',
            'timing_scope': 'raw uint8 image batch to predicted labels; includes scaling, HOG if used, pipeline scaler and model forward; excludes file I/O, model loading and score/probability generation',
            'selection_policy': 'Model configurations frozen from validation. No fitting or model selection using test scores.',
            'primary_cnn': 'cnn', 'primary_baseline': 'pixels_random_forest',
            'training_budget': 12000, 'validation_budget': 12000,
            'hog_config': HOG_CONFIG, 'bootstrap_seed': 42, 'bootstrap_repeats': 2000}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_choices(root, report):
    """Create an auditable manifest BEFORE any official test archive is opened."""
    selected = pd.read_csv(root / 'reports/baselines/validation_results.csv')
    cnn_summary = json.loads((root / 'reports/cnn/summary.json').read_text())
    baseline_summary = json.loads((root / 'reports/baselines/summary.json').read_text())
    assert f"{selected.iloc[0]['features']}_{selected.iloc[0]['model']}" == PROTOCOL['primary_baseline']
    assert baseline_summary['subset_sha256'] == cnn_summary['subset_sha256']
    assert baseline_summary['split_sha256'] == cnn_summary['split_sha256']
    assert cnn_summary['training_images'] == PROTOCOL['training_budget']
    assert cnn_summary['validation_macro_f1'] > selected.iloc[0]['macro_f1']
    records = [{'key': 'cnn', 'path': 'models/cnn/best.pt', 'features': 'cnn',
                'validation_accuracy': cnn_summary['validation_accuracy'],
                'validation_macro_f1': cnn_summary['validation_macro_f1'],
                'selection': f"minimum validation loss; epoch {cnn_summary['best_epoch']}"}]
    for row in selected.itertuples():
        key = f'{row.features}_{row.model}'
        assert row.training_images == 12000 and row.validation_images == 12000
        records.append({'key': key, 'path': f'models/baselines/{key}.joblib',
                        'features': row.features, 'parameters': json.loads(row.parameters),
                        'validation_accuracy': row.accuracy, 'validation_macro_f1': row.macro_f1,
                        'selection': 'validation macro F1; accuracy tie breaker'})
    for record in records:
        record['sha256'] = digest(root / record['path'])
    protected_paths = ['src/data.py', 'src/baselines.py', 'src/model.py', 'src/train.py',
        'data/processed/train_validation_split.npz', 'data/processed/baseline_train_indices.npy',
        'data/processed/preprocessing.json', 'notebooks/reference/LAB_12_CNN_Fashion_mnist.ipynb']
    manifest = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                'protocol': PROTOCOL, 'models': records,
                'protected_sha256': {p: digest(root / p) for p in protected_paths},
                'deployment_choice_before_test': 'cnn',
                'choice_basis': 'CNN has highest validation macro F1; primary baseline is pixel Random Forest. All nine configurations are reported; test scores will not be used for tuning.'}
    path = report / 'frozen-models.json'
    # JSON normalizes tuple-valued HOG settings to lists for stable comparisons.
    manifest = json.loads(json.dumps(manifest))
    if path.exists():
        saved = json.loads(path.read_text())
        for key in ('protocol', 'models', 'protected_sha256', 'deployment_choice_before_test', 'choice_basis'):
            if saved[key] != manifest[key]:
                raise ValueError(f'Frozen evaluation inputs differ: {key}')
        return saved
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Frozen model choices, hashes and timing protocol before loading official test data.', flush=True)
    return manifest


def load_official_test(root):
    """Download only official test archives, verify publisher MD5 and IDX headers."""
    raw = root / 'data/raw/FashionMNIST'
    raw.mkdir(parents=True, exist_ok=True)
    for name, expected in TEST_FILES.items():
        path = raw / name
        if not path.exists():
            temporary = path.with_suffix('.part')
            print(f'Downloading official test file: {name}', flush=True)
            try:
                with urllib.request.urlopen(SOURCE + name, timeout=120) as response, temporary.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if hashlib.md5(temporary.read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Checksum mismatch: {name}')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Corrupt official test file: {path}')
    with gzip.open(raw / 't10k-images-idx3-ubyte.gz', 'rb') as stream:
        assert struct.unpack('>IIII', stream.read(16)) == (2051, 10000, 28, 28)
        images = np.frombuffer(stream.read(), dtype=np.uint8).reshape(10000, 28, 28).copy()
    with gzip.open(raw / 't10k-labels-idx1-ubyte.gz', 'rb') as stream:
        assert struct.unpack('>II', stream.read(8)) == (2049, 10000)
        labels = np.frombuffer(stream.read(), dtype=np.uint8).copy()
    assert labels.shape == (10000,) and np.all(np.bincount(labels, minlength=10) == 1000)
    return images, labels


def predict_raw(model, kind, raw):
    """End-to-end in-memory raw-image-to-label path used by every timing repeat."""
    if kind == 'cnn':
        with torch.inference_mode():
            tensor = torch.from_numpy(scale_pixels(raw)[:, None, :, :])
            return model(tensor).argmax(1).numpy()
    return model.predict(features(raw, kind))


def predict_batched(model, kind, images):
    return np.concatenate([predict_raw(model, kind, images[i:i+128])
                           for i in range(0, len(images), PROTOCOL['batch_size'])])


def complexity(model, kind):
    if kind == 'cnn':
        return {'kind': 'trainable neural parameters', 'count': count_parameters(model),
                'buffer_scalars': sum(b.numel() for b in model.buffers())}
    estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
    if hasattr(estimator, 'estimators_'):
        return {'kind': 'tree nodes', 'count': sum(t.tree_.node_count for t in estimator.estimators_),
                'trees': len(estimator.estimators_),
                'max_depth': max(t.tree_.max_depth for t in estimator.estimators_)}
    if hasattr(estimator, '_fit_X'):
        return {'kind': 'stored training examples', 'count': len(estimator._fit_X),
                'stored_features': int(estimator._fit_X.shape[1]),
                'stored_feature_bytes': int(estimator._fit_X.nbytes)}
    result = {'kind': 'linear coefficients plus intercepts',
              'count': int(estimator.coef_.size + estimator.intercept_.size),
              'coefficient_count': int(estimator.coef_.size),
              'intercept_count': int(estimator.intercept_.size)}
    if hasattr(model, 'steps'):
        scaler = model.steps[0][1]
        result['scaler_mean_var_scale_scalars'] = int(scaler.mean_.size + scaler.var_.size + scaler.scale_.size)
    return result


def plot_confusion(matrices, figures):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, key, title in zip(axes, ('pixels_random_forest', 'cnn'), ('Pixel Random Forest', 'Small CNN')):
        values = matrices[key] / matrices[key].sum(axis=1, keepdims=True)
        im = ax.imshow(values, vmin=0, vmax=1, cmap='Blues')
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f'{values[i,j]*100:.0f}', ha='center', va='center',
                        fontsize=7, color='white' if values[i,j] > 0.5 else 'black')
        ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASS_NAMES,
               yticklabels=CLASS_NAMES, xlabel='Predicted label', ylabel='True label', title=title)
        plt.setp(ax.get_xticklabels(), rotation=65, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
    fig.suptitle('Official test confusion — each row sums to 100%; cells show percentages')
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    color_ax = fig.add_axes([0.94, 0.2, 0.012, 0.6])
    fig.colorbar(im, cax=color_ax, label='Fraction within true class')
    fig.savefig(figures / 'test_confusion_comparison.png', dpi=160)
    plt.close(fig)


def error_examples(images, labels, predictions, probabilities, report, figures):
    rf, cnn = predictions['pixels_random_forest'], predictions['cnn']
    groups = [('CNN correct; Random Forest wrong', (cnn == labels) & (rf != labels)),
              ('Random Forest correct; CNN wrong', (rf == labels) & (cnn != labels)),
              ('Both models wrong', (rf != labels) & (cnn != labels))]
    rng = np.random.default_rng(42)
    focus = np.isin(labels, [0, 2, 4, 6])
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    rows = []
    for row, (title, mask) in enumerate(groups):
        preferred = np.flatnonzero(mask & focus)
        extra = np.flatnonzero(mask & ~focus)
        chosen = list(rng.permutation(preferred)[:4])
        if len(chosen) < 4:
            chosen += list(rng.permutation(extra)[:4-len(chosen)])
        for col, ax in enumerate(axes[row]):
            ax.axis('off')
            if col >= len(chosen):
                continue
            idx = int(chosen[col])
            ax.imshow(images[idx], cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'True: {CLASS_NAMES[labels[idx]]}\nRF: {CLASS_NAMES[rf[idx]]}\nCNN: {CLASS_NAMES[cnn[idx]]}', fontsize=10)
            rows.append({'group': title, 'test_index': idx, 'true_label': int(labels[idx]),
                         'rf_prediction': int(rf[idx]), 'cnn_prediction': int(cnn[idx]),
                         'cnn_max_softmax': float(probabilities[idx].max())})
        axes[row, 0].text(-0.18, 0.5, title, rotation=90, va='center', ha='center',
                          transform=axes[row, 0].transAxes, fontsize=10)
    fig.suptitle('Illustrative test examples — fixed-seed samples prioritizing upper-body categories')
    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    fig.savefig(figures / 'test_error_examples.png', dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(report / 'error_examples.csv', index=False)


def run(root):
    root = Path(root).resolve()
    report = root / 'reports/final_evaluation'
    figures = root / 'reports/figures'
    report.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    if (report / 'summary.json').exists():
        raise FileExistsError('Final evaluation already complete; reuse saved results instead of retuning on test data.')
    torch.set_num_threads(PROTOCOL['library_threads'])
    manifest = freeze_choices(root, report)
    images, labels = load_official_test(root)
    print('Official test:', images.shape, '; 1000 examples per class.', flush=True)
    all_metrics, all_timing, recalls, sizes = [], [], [], {}
    predictions, matrices, probabilities = {}, {}, None
    for record in manifest['models']:
        key, kind = record['key'], record['features']
        assert digest(root / record['path']) == record['sha256']
        if kind == 'cnn':
            checkpoint = torch.load(root / record['path'], map_location='cpu', weights_only=True)
            assert checkpoint['class_names'] == list(CLASS_NAMES)
            model = SmallCNN()
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
        else:
            model = joblib.load(root / record['path'])
            assert np.array_equal(model.classes_, np.arange(10))
        print(f'Evaluating frozen {key} on CPU...', flush=True)
        pred = predict_batched(model, kind, images)
        assert pred.shape == labels.shape and np.isin(pred, np.arange(10)).all()
        predictions[key] = pred
        matrices[key] = confusion_matrix(labels, pred, labels=np.arange(10))
        np.savetxt(report / f'{key}_confusion.csv', matrices[key], delimiter=',', fmt='%d')
        if kind == 'cnn':
            with torch.inference_mode():
                probabilities = np.concatenate([model(torch.from_numpy(scale_pixels(images[i:i+128])[:, None])).softmax(1).numpy()
                    for i in range(0, len(images), 128)])
            assert np.allclose(probabilities.sum(1), 1, atol=1e-5)
            assert np.array_equal(probabilities.argmax(1), pred)
        batch = images[:PROTOCOL['timing_images']]
        predict_raw(model, kind, batch[:128])
        repeats = []
        for repeat in range(PROTOCOL['timing_repeats']):
            start = time.perf_counter()
            timed = predict_batched(model, kind, batch)
            seconds = time.perf_counter() - start
            assert np.array_equal(timed, pred[:len(batch)])
            repeats.append(seconds)
            all_timing.append({'model': key, 'repeat': repeat + 1, 'images': len(batch),
                               'batch_size': 128, 'seconds': seconds, 'ms_per_image': seconds * 1000 / len(batch)})
        info = complexity(model, kind)
        sizes[key] = info
        all_metrics.append({'model': key, 'test_accuracy': accuracy_score(labels, pred),
            'test_macro_f1': f1_score(labels, pred, average='macro'),
            'validation_accuracy': record['validation_accuracy'],
            'validation_macro_f1': record['validation_macro_f1'],
            'training_images': 12000, 'test_images': len(labels),
            'cpu_ms_per_image_median': float(np.median(repeats) * 1000 / len(batch)),
            'cpu_images_per_second': float(len(batch) / np.median(repeats)),
            'saved_model_bytes': (root / record['path']).stat().st_size,
            'complexity_kind': info['kind'], 'complexity_count': info['count']})
        for label, recall in enumerate(recall_score(labels, pred, labels=np.arange(10), average=None)):
            recalls.append({'model': key, 'label': label, 'category': CLASS_NAMES[label], 'recall': recall})
        pd.DataFrame(all_metrics).to_csv(report / 'test_results.csv', index=False)
        print(f"  test accuracy={all_metrics[-1]['test_accuracy']:.4f}; macro F1={all_metrics[-1]['test_macro_f1']:.4f}; "
              f"CPU end-to-end {all_metrics[-1]['cpu_ms_per_image_median']:.3f} ms/image", flush=True)
        del model
    pd.DataFrame(all_timing).to_csv(report / 'timing_repeats.csv', index=False)
    pd.DataFrame(recalls).to_csv(report / 'per_class_recall.csv', index=False)
    (report / 'model_complexity.json').write_text(json.dumps(sizes, indent=2) + '\n')
    np.savez_compressed(report / 'test_predictions.npz', test_indices=np.arange(len(labels)),
                        true_labels=labels, cnn_probabilities=probabilities, **predictions)
    plot_confusion(matrices, figures)
    error_examples(images, labels, predictions, probabilities, report, figures)
    recall_table = pd.DataFrame(recalls).pivot(index='category', columns='model', values='recall').reindex(CLASS_NAMES)
    ax = recall_table[['pixels_random_forest', 'cnn']].plot.bar(figsize=(12, 5), ylim=(0, 1), rot=35)
    ax.set(xlabel='True clothing category', ylabel='Official test recall', title='Per-class test recall — models selected before test evaluation')
    ax.grid(axis='y', alpha=0.2)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(figures / 'test_per_class_recall.png', dpi=150)
    plt.close(ax.get_figure())
    cnn_correct = predictions['cnn'] == labels
    rf_correct = predictions['pixels_random_forest'] == labels
    overlap = {'both_correct': int(np.sum(cnn_correct & rf_correct)),
               'cnn_only_correct': int(np.sum(cnn_correct & ~rf_correct)),
               'rf_only_correct': int(np.sum(~cnn_correct & rf_correct)),
               'both_wrong': int(np.sum(~cnn_correct & ~rf_correct))}
    diff = cnn_correct.astype(np.int8) - rf_correct.astype(np.int8)
    rng = np.random.default_rng(PROTOCOL['bootstrap_seed'])
    resampled = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                          for _ in range(PROTOCOL['bootstrap_repeats'])])
    ci = np.percentile(resampled, [2.5, 97.5]).tolist()
    confusion_rows = []
    for key in ('pixels_random_forest', 'cnn'):
        for true in range(10):
            for predicted in range(10):
                if true != predicted:
                    confusion_rows.append({'model': key, 'true': CLASS_NAMES[true],
                        'predicted': CLASS_NAMES[predicted], 'count': int(matrices[key][true, predicted]),
                        'upper_body_pair': true in [0,2,4,6] and predicted in [0,2,4,6]})
    confusions = pd.DataFrame(confusion_rows).sort_values(['model', 'count'], ascending=[True, False])
    confusions.to_csv(report / 'confusion_pairs.csv', index=False)
    metrics = pd.DataFrame(all_metrics).set_index('model')
    cnn_row, rf_row = metrics.loc['cnn'], metrics.loc['pixels_random_forest']
    summary = {'milestone': 5, 'status': 'complete', 'official_test_images': len(labels),
               'official_test_md5': TEST_FILES, 'model_count': len(all_metrics),
               'frozen_manifest_sha256': digest(report / 'frozen-models.json'),
               'chosen_model_before_test': 'cnn', 'primary_baseline_before_test': 'pixels_random_forest',
               'cnn_test_accuracy': float(cnn_row.test_accuracy), 'cnn_test_macro_f1': float(cnn_row.test_macro_f1),
               'baseline_test_accuracy': float(rf_row.test_accuracy), 'baseline_test_macro_f1': float(rf_row.test_macro_f1),
               'accuracy_difference': float(diff.mean()), 'paired_bootstrap_95ci_accuracy_difference': ci,
               'bootstrap_scope': 'Test-image sampling uncertainty for these fixed fitted models; excludes training/seed variability.',
               'paired_correctness': overlap, 'timing_protocol': PROTOCOL,
               'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                               'torch': str(torch.__version__), 'numpy': np.__version__,
                               'scikit_learn': sklearn.__version__, 'scikit_image': skimage.__version__},
               'limitations': ['Single trained seed and 12000-image training subset for every model.',
                              'Different optimization/search budgets and checkpoint selection criteria.',
                              'Timing uses resident arrays, warm models, batch size 128 and CPU only; not upload latency or single-image latency.',
                              'Serialized sizes use different formats; tree nodes, stored samples and neural parameters are not equivalent.',
                              'Fashion-MNIST grayscale thumbnails do not establish performance on real product photos.',
                              'Official test results are final reporting; further changes require new validation-based decisions, not test-driven tuning.']}
    comparison_lines = ['# Final official-test evaluation', '',
        'All nine model configurations and the primary CNN/Random Forest comparison were frozen before official test loading. No fitting or hyperparameter search occurred in this milestone.', '',
        f"CNN: accuracy **{cnn_row.test_accuracy:.2%}**, macro F1 **{cnn_row.test_macro_f1:.4f}**. Pixel Random Forest: accuracy **{rf_row.test_accuracy:.2%}**, macro F1 **{rf_row.test_macro_f1:.4f}**.", '',
        f"Accuracy difference: **{diff.mean()*100:.2f} percentage points**. Paired test-image bootstrap 95% interval: **{ci[0]*100:.2f} to {ci[1]*100:.2f} points** (2,000 resamples, seed 42). This interval does not include training-seed variability.", '',
        f"CNN correct/RF wrong: {overlap['cnn_only_correct']}; RF correct/CNN wrong: {overlap['rf_only_correct']}; both wrong: {overlap['both_wrong']}; both correct: {overlap['both_correct']}.", '',
        '## Per-class recall', '', '| Category | Pixel Random Forest | CNN |', '|---|---:|---:|']
    for label in CLASS_NAMES:
        comparison_lines.append(f"| {label} | {recall_table.loc[label, 'pixels_random_forest']:.1%} | {recall_table.loc[label, 'cnn']:.1%} |")
    comparison_lines += ['', '## Frequent upper-body confusions', '']
    for key in ('pixels_random_forest', 'cnn'):
        top = confusions[(confusions.model == key) & confusions.upper_body_pair].head(5)
        comparison_lines.append(f"{key}: " + '; '.join(f'{r.true} → {r.predicted}: {r.count}' for r in top.itertuples()) + '.')
    comparison_lines += ['', '## Interpretation and limits', '',
        'The paired outcomes show where learned spatial features recover baseline errors and where the CNN still fails. Examples in the figure are fixed-seed illustrations prioritizing shirts, T-shirts/tops, pullovers and coats; they are not a representative sample of all errors.', '',
        'CPU timing includes descriptor extraction and fitted preprocessing, with identical image batches and three warmed repetitions. Raw repeats are saved. It excludes decoding, file access and model loading. HOG must be computed for new images and its cost is included.', '',
        'The complexity report uses weights for linear models, tree nodes for forests, stored examples/features for kNN, and trainable parameters for the CNN. These quantities are not directly equivalent. File sizes reflect the actual saved serialization format.', '',
        *[f'- {limitation}' for limitation in summary['limitations']], '',
        'The deployment choice remains the CNN selected using validation before test evaluation. Next milestone: reusable upload-image preprocessing and prediction, matching the saved input contract.']
    (report / 'analysis.md').write_text('\n'.join(comparison_lines) + '\n')
    # Final integrity and score checks are independent of the in-memory estimator objects.
    for record in manifest['models']:
        assert digest(root / record['path']) == record['sha256']
    for path, expected in manifest['protected_sha256'].items():
        assert digest(root / path) == expected
    with np.load(report / 'test_predictions.npz', allow_pickle=False) as saved:
        for key in predictions:
            assert np.isclose(accuracy_score(saved['true_labels'], saved[key]), metrics.loc[key, 'test_accuracy'])
            assert np.isclose(f1_score(saved['true_labels'], saved[key], average='macro'), metrics.loc[key, 'test_macro_f1'])
            assert matrices[key].sum() == 10000
    (report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"Final primary comparison: CNN {cnn_row.test_accuracy:.2%}, pixel Random Forest {rf_row.test_accuracy:.2%}; "
          f"difference {diff.mean()*100:.2f} percentage points. All model hashes unchanged.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    with threadpool_limits(limits=PROTOCOL['library_threads']):
        run(args.project_root)

if __name__ == '__main__':
    main()
