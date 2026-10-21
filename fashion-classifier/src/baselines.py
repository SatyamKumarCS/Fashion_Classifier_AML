"""Budgeted pixel/HOG baselines. Never load the official test set.

Run from the project root: python -m src.baselines
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import skimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.exceptions import ConvergenceWarning
from skimage.feature import hog
from threadpoolctl import threadpool_limits

from .data import CLASS_NAMES, load_training_data, load_or_create_split, scale_pixels

HOG_CONFIG = dict(orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2),
                  block_norm='L2-Hys', transform_sqrt=False, feature_vector=True)
SEED = 42
TRAIN_BUDGET = 12000


def features(images, kind):
    """Fixed image descriptors; no fitting or validation-label access."""
    if kind == 'pixels':
        return scale_pixels(images).reshape(len(images), -1)
    if kind != 'hog':
        raise ValueError(kind)
    result = np.empty((len(images), 324), dtype=np.float32)
    for i, image in enumerate(images):
        result[i] = hog(scale_pixels(image), **HOG_CONFIG)
        if (i + 1) % 10000 == 0:
            print(f'HOG: {i+1}/{len(images)} images', flush=True)
    return result


def candidates():
    """Small predeclared search, identical for both feature representations."""
    return {
        'logistic_regression': [
            ({'C': c, 'max_iter': 1000, 'solver': 'lbfgs'},
             make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000,
                           solver='lbfgs', random_state=SEED))) for c in (0.1, 1.0)],
        'knn': [({'n_neighbors': k, 'weights': 'distance', 'metric': 'euclidean'},
                 KNeighborsClassifier(n_neighbors=k, weights='distance',
                                      metric='euclidean', n_jobs=2)) for k in (3, 5)],
        'random_forest': [({'n_estimators': 100, 'max_depth': depth, 'max_features': 'sqrt'},
                           RandomForestClassifier(n_estimators=100, max_depth=depth,
                           max_features='sqrt', random_state=SEED, n_jobs=2))
                          for depth in (20, None)],
        'linear_svm': [({'C': c, 'max_iter': 10000, 'dual': 'auto'},
                        make_pipeline(StandardScaler(), LinearSVC(C=c, dual='auto',
                        max_iter=10000, random_state=SEED))) for c in (0.1, 1.0)],
    }


def run(project_root):
    root = Path(project_root).resolve()
    report = root / 'reports/baselines'
    report.mkdir(parents=True, exist_ok=True)
    model_dir = root / 'models/baselines'
    model_dir.mkdir(parents=True, exist_ok=True)
    if (report / 'summary.json').exists():
        raise FileExistsError('Baseline run already complete; review saved outputs before rerunning.')
    images, labels = load_training_data(root, download=False)
    train_idx, val_idx = load_or_create_split(labels, root)
    subset, _ = train_test_split(train_idx, train_size=TRAIN_BUDGET, random_state=SEED,
                                stratify=labels[train_idx])
    assert not np.intersect1d(subset, val_idx).size
    assert np.all(np.bincount(labels[subset], minlength=10) == 1200)
    subset_path = root / 'data/processed/baseline_train_indices.npy'
    if subset_path.exists():
        if not np.array_equal(np.load(subset_path, allow_pickle=False), subset):
            raise ValueError('Saved baseline subset does not match this run')
    else:
        np.save(subset_path, subset)
    all_rows, best_rows, predictions = [], [], {}
    for kind in ('pixels', 'hog'):
        start = time.perf_counter()
        # Fixed descriptors are computed only for the selected training and validation rows.
        x_train, x_val = features(images[subset], kind), features(images[val_idx], kind)
        feature_seconds = time.perf_counter() - start
        for family, options in candidates().items():
            best_key, best = None, None
            for params, estimator in options:
                print(f'Training {kind}/{family}: {params}', flush=True)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    start = time.perf_counter()
                    estimator.fit(x_train, labels[subset])
                    fit_seconds = time.perf_counter() - start
                convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                start = time.perf_counter()
                pred = estimator.predict(x_val)
                predict_seconds = time.perf_counter() - start
                row = dict(features=kind, model=family, parameters=json.dumps(params, sort_keys=True),
                           training_images=len(subset), validation_images=len(val_idx),
                           feature_count=x_train.shape[1], accuracy=accuracy_score(labels[val_idx], pred),
                           macro_f1=f1_score(labels[val_idx], pred, average='macro'),
                           fit_seconds=fit_seconds, validation_predict_seconds=predict_seconds,
                           feature_extraction_seconds=feature_seconds,
                           convergence_warnings='; '.join(convergence))
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(report / 'candidates.csv', index=False)
                print(f"  validation accuracy={row['accuracy']:.4f}, macro F1={row['macro_f1']:.4f}; fit={fit_seconds:.1f}s", flush=True)
                if convergence:
                    print('  Convergence warning: candidate excluded from selection', flush=True)
                    continue
                key = (row['macro_f1'], row['accuracy'])
                if best_key is None or key > best_key:
                    best_key, best = key, (row, estimator, pred)
            if best is None:
                raise RuntimeError(f'No converged candidates for {kind}/{family}; inspect candidates.csv')
            row, estimator, pred = best
            row = dict(row)
            model_path = model_dir / f'{kind}_{family}.joblib'
            joblib.dump(estimator, model_path, compress=3)
            row['saved_model_bytes'] = model_path.stat().st_size
            best_rows.append(row)
            predictions[f'{kind}_{family}'] = pred
            pd.DataFrame(best_rows).to_csv(report / 'validation_results.csv', index=False)
            per_class = pd.DataFrame({'label': np.arange(10), 'category': CLASS_NAMES,
                'validation_recall': recall_score(labels[val_idx], pred, average=None, labels=np.arange(10))})
            per_class.to_csv(report / f'{kind}_{family}_recall.csv', index=False)
        del x_train, x_val
    np.savez_compressed(report / 'validation_predictions.npz', validation_indices=val_idx,
                        true_labels=labels[val_idx], **predictions)
    ranked = pd.DataFrame(best_rows).sort_values(['macro_f1', 'accuracy'], ascending=False)
    ranked.to_csv(report / 'validation_results.csv', index=False)
    best = ranked.iloc[0]
    summary = dict(milestone=3, status='complete', seed=SEED, training_budget=TRAIN_BUDGET,
        full_training_split_size=len(train_idx), validation_size=len(val_idx),
        subset_per_class=1200, feature_config={'pixels': {'shape': [784], 'scaling': 'float32 / 255'},
        'hog': {**HOG_CONFIG, 'feature_count': 324}},
        learned_preprocessing='StandardScaler fitted on baseline training subset only for LR and SVM; none for kNN and RF',
        selection='highest validation macro F1; accuracy breaks ties; converged candidates only',
        candidate_count=len(all_rows), selected_model_count=len(best_rows),
        best_validation_model=f"{best['features']}/{best['model']}",
        best_validation_macro_f1=float(best['macro_f1']),
        best_validation_accuracy=float(best['accuracy']),
        split_sha256=hashlib.sha256((root / 'data/processed/train_validation_split.npz').read_bytes()).hexdigest(),
        subset_sha256=hashlib.sha256(subset.tobytes()).hexdigest(),
        versions={'numpy': np.__version__, 'scikit_learn': sklearn.__version__, 'scikit_image': skimage.__version__},
        timing='wall clock on local CPU with 2 threads; validation prediction includes pipeline transforms, excludes feature extraction; preliminary, not final inference benchmark',
        limitation='All baselines use a shared 12000-image subset. These are validation selection scores, not official test scores or a full-training-data comparison.',
        official_test_loaded=False)
    (report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(ranked[['features', 'model', 'accuracy', 'macro_f1']].to_string(index=False), flush=True)
    return ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.project_root)

if __name__ == '__main__':
    main()
