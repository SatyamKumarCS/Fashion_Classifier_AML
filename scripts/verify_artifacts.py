"""Verify every hash-protected artifact and report which model weights are installed.

Run from the project root: python scripts/verify_artifacts.py
- Tracked files named in the committed manifests (source, reports, reference notebook) must match.
- data/processed, when present, must match the original split and preprocessing hashes.
- models/ must be either the original frozen artifacts or a reproduction recorded in
  models/PROVENANCE.json whose hashes still match. Unrecorded or altered models fail.
Reads files only; never loads datasets. Prints JSON and exits 1 on any failure.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ['reports/demo/protected-sha256.json', 'reports/inference/protected-sha256.json']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_hashes(root):
    """Union of all committed manifests; they must agree wherever they overlap."""
    frozen = json.loads((root / 'reports/final_evaluation/frozen-models.json').read_text())
    sources = [json.loads((root / m).read_text()) for m in MANIFESTS]
    sources += [frozen['protected_sha256'], {m['path']: m['sha256'] for m in frozen['models']}]
    expected = {}
    for source in sources:
        for path, digest in source.items():
            if expected.setdefault(path, digest) != digest:
                raise ValueError(f'Committed manifests disagree about {path}')
    return expected


def check(root=ROOT):
    expected = expected_hashes(root)
    failures, report = [], {}
    tracked = {p: h for p, h in expected.items() if not p.startswith(('models/', 'data/'))}
    bad = [p for p, h in tracked.items() if not (root / p).is_file() or sha256(root / p) != h]
    failures += [f'tracked artifact changed or missing: {p}' for p in bad]
    report['tracked'] = f'{len(tracked) - len(bad)}/{len(tracked)} match committed hashes'

    data = {p: h for p, h in expected.items() if p.startswith('data/') and not p.endswith('.gitkeep')}
    present = [p for p in data if (root / p).is_file()]
    bad = [p for p in present if sha256(root / p) != data[p]]
    failures += [f'data file differs from original: {p}' for p in bad]
    report['data_processed'] = 'absent' if not present else f'{len(present) - len(bad)}/{len(data)} present and matching'

    models = {p: h for p, h in expected.items() if p.startswith('models/') and not p.endswith('.gitkeep')}
    installed = {p: sha256(root / p) for p in models if (root / p).is_file()}
    provenance_path = root / 'models/PROVENANCE.json'
    if not installed:
        report['models'] = 'absent'
    elif installed == models:
        report['models'] = 'original frozen artifacts (match reports/final_evaluation)'
    elif provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        recorded = provenance['sha256']
        changed = [p for p, h in recorded.items() if not (root / p).is_file() or sha256(root / p) != h]
        unrecorded = [p for p in installed if p not in recorded]
        failures += [f'model differs from models/PROVENANCE.json: {p}' for p in changed]
        failures += [f'model not recorded in models/PROVENANCE.json: {p}' for p in unrecorded]
        report['models'] = (f"reproduction from {provenance['run_dir']} (not the committed checkpoints); "
                            f"CNN test accuracy {provenance['this_run_test_accuracy']['cnn']:.4f} vs committed "
                            f"{provenance['committed_test_accuracy']['cnn']:.4f}")
    else:
        failures.append('models/ differ from the frozen artifacts and have no models/PROVENANCE.json')
        report['models'] = 'unrecorded'
    report['status'] = 'failed' if failures else 'passed'
    report['failures'] = failures
    return report


def main():
    report = check()
    print(json.dumps(report, indent=2))
    return 1 if report['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
