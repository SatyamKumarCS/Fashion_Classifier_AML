"""reproduce.py installs never overwrite different artifacts and never copy test archives."""
from conftest import load_script

reproduce = load_script('reproduce')


def make_run(tmp_path, weights=b'weights'):
    run = tmp_path / 'run'
    for relative, content in {'models/cnn/best.pt': weights, 'models/baselines/m.joblib': b'model'}.items():
        (run / relative).parent.mkdir(parents=True, exist_ok=True)
        (run / relative).write_bytes(content)
    raw = run / 'data/raw/FashionMNIST'
    raw.mkdir(parents=True)
    for name in [*reproduce.data.TRAIN_FILES, 't10k-images-idx3-ubyte.gz']:
        (raw / name).write_bytes(name.encode())
    (run / 'data/processed').mkdir(parents=True)
    for name in reproduce.PROCESSED:
        (run / 'data/processed' / name).write_bytes(name.encode())
    return run


def test_install_copies_training_data_only_and_records(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(reproduce, 'record_provenance', lambda run, root: recorded.append(root))
    run, root = make_run(tmp_path), tmp_path / 'project'
    reproduce.install(run, root)
    assert (root / 'models/cnn/best.pt').read_bytes() == b'weights'
    assert not list(root.rglob('t10k*'))
    assert all((root / 'data/raw/FashionMNIST' / n).exists() for n in reproduce.data.TRAIN_FILES)
    assert recorded == [root]
    reproduce.install(run, root)  # idempotent: same files are recognised, provenance re-checked
    assert recorded == [root, root]


def test_install_keeps_different_models_without_provenance(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(reproduce, 'record_provenance', lambda run, root: recorded.append(root))
    root = tmp_path / 'project'
    (root / 'models/cnn').mkdir(parents=True)
    (root / 'models/cnn/best.pt').write_bytes(b'original')
    reproduce.install(make_run(tmp_path), root)
    assert (root / 'models/cnn/best.pt').read_bytes() == b'original'
    assert recorded == []
