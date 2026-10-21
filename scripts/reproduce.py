"""Run the full pipeline (data -> baselines -> CNN -> frozen test evaluation) in a clean root.

Run from the project root: python scripts/reproduce.py --install
The committed reports/ are protected against overwriting, so a fresh run writes into
--run-dir and leaves them untouched. Completed stages in --run-dir are skipped on rerun;
delete the run directory after a failed stage to start over.

With --install, the run's models are copied into models/ (never replacing different models),
and its training archives and split files into data/ (training files only; never the test set).
Installed models are recorded in models/PROVENANCE.json because they are a reproduction,
not the checkpoint behind the committed results; models/reproduced-sha256.json is the
matching integrity manifest used by scripts/verify_demo.py and scripts/verify_artifacts.py.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import sys

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import baselines, data, evaluate, train
from src.predict import INFERENCE_CONFIG

# evaluate.freeze_choices fingerprints these, so the run root needs its own copies.
COPIED = ['src', 'notebooks/reference/LAB_12_CNN_Fashion_mnist.ipynb']
PROCESSED = ['train_validation_split.npz', 'baseline_train_indices.npy', 'preprocessing.json']
ORIGINAL_MANIFEST = 'reports/demo/protected-sha256.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(run):
    for relative in COPIED:
        source, target = ROOT / relative, run / relative
        if target.exists():
            continue  # keep the snapshot of the code that produced this run
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            shutil.copy2(source, target)
    # Same steps the notebook performs before Step 3; training archives only.
    _, labels = data.load_training_data(run, download=True)
    data.load_or_create_split(labels, run)
    data.save_preprocessing_config(run)


def stage(name, done, action):
    if done.exists():
        print(f'== {name}: already complete in run dir, skipping', flush=True)
        return
    print(f'== {name}', flush=True)
    action()


def same_files(source, target):
    files = [p.relative_to(source) for p in source.rglob('*') if p.is_file()]
    return all((target / f).is_file() and sha256(source / f) == sha256(target / f) for f in files)


def copy_if_absent(source, target):
    """Copy a file unless the target exists; an existing different file is kept and reported."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return 'copied'
    return 'present' if sha256(source) == sha256(target) else 'kept-different'


def install(run, root=ROOT):
    from_run = []
    for relative in ('models/cnn', 'models/baselines'):
        source, target = run / relative, root / relative
        if target.exists() and any(target.iterdir()):
            if same_files(source, target):
                from_run.append(relative)
                print(f'{relative} is already installed from this run.', flush=True)
            else:
                print(f'Kept existing {relative}; it differs from this run and was not overwritten.', flush=True)
            continue
        shutil.copytree(source, target, dirs_exist_ok=True)
        from_run.append(relative)
        print(f'Installed {relative} for app.py.', flush=True)
    raw = 'data/raw/FashionMNIST'
    copies = {f'{raw}/{name}': copy_if_absent(run / raw / name, root / raw / name) for name in data.TRAIN_FILES}
    copies.update({f'data/processed/{name}': copy_if_absent(run / 'data/processed' / name, root / 'data/processed' / name)
                   for name in PROCESSED})
    for path, status in copies.items():
        if status == 'kept-different':
            print(f'WARNING: kept existing {path}; it differs from this run.', flush=True)
    print('Training data for verification:', sum(s == 'copied' for s in copies.values()), 'copied,',
          sum(s == 'present' for s in copies.values()), 'already present.', flush=True)
    if len(from_run) == 2:
        record_provenance(run, root)
    else:
        print('Provenance not recorded: models/ is not entirely from this run.', flush=True)


def record_provenance(run, root):
    """Describe installed models; written once per installation and never for unrelated models."""
    provenance_path = root / 'models/PROVENANCE.json'
    if provenance_path.exists():
        print('models/PROVENANCE.json already present; left unchanged.', flush=True)
        return
    checkpoint = root / 'models/cnn/best.pt'
    contract = root / 'models/cnn/inference-config.json'
    if not contract.exists():
        contract.write_text(json.dumps({**INFERENCE_CONFIG, 'class_names': list(data.CLASS_NAMES),
                                        'checkpoint': 'models/cnn/best.pt', 'checkpoint_sha256': sha256(checkpoint),
                                        'provenance': 'models/PROVENANCE.json'}, indent=2) + '\n')
    frozen = json.loads((root / 'reports/final_evaluation/frozen-models.json').read_text())
    original = {m['path']: m['sha256'] for m in frozen['models']}
    hashes = {str(p.relative_to(root)): sha256(p) for d in ('models/cnn', 'models/baselines')
              for p in sorted((root / d).rglob('*')) if p.is_file()}
    run_results = pd.read_csv(run / 'reports/final_evaluation/test_results.csv').set_index('model')
    committed = pd.read_csv(root / 'reports/final_evaluation/test_results.csv').set_index('model')
    trained = json.loads((run / 'models/cnn/config.json').read_text())
    provenance = {
        'source': 'scripts/reproduce.py (Docker reproduction)',
        'run_dir': os.path.relpath(run, root),
        'installed_utc': datetime.now(timezone.utc).isoformat(),
        'matches_original_frozen_models': all(hashes.get(p) == h for p, h in original.items()),
        'note': ('Retrained with the committed code and identical data split. These are not the checkpoints '
                 'behind the committed reports (reports/final_evaluation); those weights are not in this '
                 'repository. Results below come from this run\'s own single frozen-model test evaluation.'),
        'cnn_training_environment': {k: trained[k] for k in ('device', 'torch', 'python', 'platform', 'numpy')},
        'this_run_test_accuracy': run_results['test_accuracy'].round(4).to_dict(),
        'committed_test_accuracy': committed['test_accuracy'].round(4).to_dict(),
        'sha256': hashes,
        'original_frozen_sha256': original,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    manifest = json.loads((root / ORIGINAL_MANIFEST).read_text())
    reproduced = {path: sha256(root / path) if path.startswith('models/') and (root / path).exists() else expected
                  for path, expected in manifest.items()}
    (root / 'models/reproduced-sha256.json').write_text(json.dumps(reproduced, indent=2) + '\n')
    print('Recorded models/PROVENANCE.json and models/reproduced-sha256.json '
          f"(this run's CNN test accuracy: {provenance['this_run_test_accuracy']['cnn']:.4f}).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/reproduce')
    parser.add_argument('--install', action='store_true', help='Copy trained models and training data into the project.')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    print(f'Run directory: {run}', flush=True)
    prepare(run)
    with threadpool_limits(limits=2):
        stage('Baselines', run / 'reports/baselines/summary.json', lambda: baselines.run(run))
    stage('CNN training', run / 'reports/cnn/summary.json', lambda: train.run(run))
    with threadpool_limits(limits=evaluate.PROTOCOL['library_threads']):
        stage('Final evaluation', run / 'reports/final_evaluation/summary.json', lambda: evaluate.run(run))
    if args.install:
        install(run)
    results = pd.read_csv(run / 'reports/final_evaluation/test_results.csv')
    print('\nOfficial test results (this run):')
    print(results[['model', 'test_accuracy', 'test_macro_f1', 'validation_accuracy']].to_string(index=False))


if __name__ == '__main__':
    main()
