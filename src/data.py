"""Training-only Fashion-MNIST loading and a persistent shared split."""
from pathlib import Path
import gzip
import hashlib
import json
import struct
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split

CLASS_NAMES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
SOURCE = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
TRAIN_FILES = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
}


def load_training_data(project_root, download=False):
    """Return uint8 images (60000,28,28) and labels. Never fetch test files.

    Downloads are atomic and checked against the publisher's MD5 values.
    Existing corrupt archives raise an error instead of being overwritten.
    """
    raw = Path(project_root) / 'data/raw/FashionMNIST'
    for name, checksum in TRAIN_FILES.items():
        path = raw / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path}: run once with download=True')
            raw.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.part')
            try:
                print(f'Downloading training file: {name}', flush=True)
                with urllib.request.urlopen(SOURCE + name, timeout=120) as response:
                    with temporary.open('wb') as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                if hashlib.md5(temporary.read_bytes()).hexdigest() != checksum:
                    raise ValueError(f'Checksum mismatch: {name}')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.md5(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Corrupt training archive: {path}')
    with gzip.open(raw / 'train-images-idx3-ubyte.gz', 'rb') as stream:
        header = struct.unpack('>IIII', stream.read(16))
        if header != (2051, 60000, 28, 28):
            raise ValueError(f'Unexpected image header: {header}')
        images = np.frombuffer(stream.read(), dtype=np.uint8).reshape(60000, 28, 28).copy()
    with gzip.open(raw / 'train-labels-idx1-ubyte.gz', 'rb') as stream:
        header = struct.unpack('>II', stream.read(8))
        if header != (2049, 60000):
            raise ValueError(f'Unexpected label header: {header}')
        labels = np.frombuffer(stream.read(), dtype=np.uint8).copy()
    if labels.shape != (60000,) or not np.array_equal(np.unique(labels), np.arange(10)):
        raise ValueError('Unexpected labels')
    return images, labels


def load_or_create_split(labels, project_root):
    """Save/reuse seed-42 stratified indices; reject incompatible saved files.

    Indices refer to the official training data in its original order.
    No learned preprocessing is fitted at this stage.
    """
    labels = np.asarray(labels)
    if labels.shape != (60000,) or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 6000)):
        raise ValueError('Expected all 60,000 balanced official training labels')
    path = Path(project_root) / 'data/processed/train_validation_split.npz'
    expected_train, expected_val = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels)
    fingerprint = hashlib.sha256(labels.astype(np.uint8).tobytes()).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            train_idx, val_idx = saved['train_indices'], saved['validation_indices']
            compatible = (
                np.array_equal(train_idx, expected_train)
                and np.array_equal(val_idx, expected_val)
                and saved['labels_sha256'].item() == fingerprint
                and saved['seed'].item() == 42
                and saved['validation_fraction'].item() == 0.2)
        if not compatible:
            raise ValueError(f'Incompatible saved split; preserve and inspect {path}')
    else:
        train_idx, val_idx = expected_train, expected_val
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, train_indices=train_idx, validation_indices=val_idx,
                            seed=42, validation_fraction=0.2, labels_sha256=fingerprint)
    assert len(train_idx) == 48000 and len(val_idx) == 12000
    assert np.intersect1d(train_idx, val_idx).size == 0
    assert np.array_equal(np.sort(np.concatenate([train_idx, val_idx])), np.arange(60000))
    assert np.all(np.bincount(labels[train_idx], minlength=10) == 4800)
    assert np.all(np.bincount(labels[val_idx], minlength=10) == 1200)
    return train_idx, val_idx


def scale_pixels(images):
    """Map uint8 pixels to float32 [0,1], preserving shape and orientation."""
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise TypeError('Expected raw uint8 pixels; avoid scaling twice')
    return images.astype(np.float32) / np.float32(255)


def save_preprocessing_config(project_root):
    """Record the fixed input contract; refuse to replace different settings."""
    config = {
        'dataset': 'Fashion-MNIST', 'source': SOURCE,
        'training_archive_md5': TRAIN_FILES,
        'image_shape': [28, 28], 'raw_dtype': 'uint8',
        'output_dtype': 'float32', 'pixel_scaling': 'x / 255.0',
        'output_range': [0, 1], 'orientation': 'original; no transpose or rotation',
        'background': 'original dark background; no inversion',
        'class_names': list(CLASS_NAMES), 'split_seed': 42,
        'train_size': 48000, 'validation_size': 12000,
        'split_file': 'data/processed/train_validation_split.npz',
        'learned_preprocessing': 'none; fit future scalers on training indices only',
        'official_test_policy': 'reserved for final evaluation; not loaded in Step 2',
    }
    path = Path(project_root) / 'data/processed/preprocessing.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f'Different preprocessing config already exists: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + '\n')
    return config
