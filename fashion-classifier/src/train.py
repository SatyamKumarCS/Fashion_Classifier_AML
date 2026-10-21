"""Train a budget-matched CNN with validation-only early stopping.

Run from project root: python -m src.train
No downloads and no official test loading are performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from .data import CLASS_NAMES, load_training_data, load_or_create_split, scale_pixels
from .model import SmallCNN, MODEL_CONFIG, count_parameters

TRAIN_CONFIG = {'seed': 42, 'batch_size': 128, 'max_epochs': 25, 'patience': 5,
                'min_delta': 0.0, 'optimizer': 'Adam', 'learning_rate': 0.001,
                'weight_decay': 0.0001, 'training_budget': 12000,
                'selection_metric': 'minimum validation cross-entropy loss',
                'augmentation': 'none', 'num_workers': 0, 'cpu_threads': 2}


def make_loader(images, labels, indices, shuffle=False):
    x = torch.from_numpy(scale_pixels(images[indices])[:, None, :, :])
    y = torch.from_numpy(labels[indices].astype(np.int64))
    generator = torch.Generator().manual_seed(TRAIN_CONFIG['seed'])
    return DataLoader(TensorDataset(x, y), batch_size=TRAIN_CONFIG['batch_size'],
                      shuffle=shuffle, num_workers=0, generator=generator)


def evaluate(model, loader, device):
    """No gradients or BatchNorm updates; return sample-weighted mean loss."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    probabilities, truth = [], []
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
            probabilities.append(logits.softmax(1).cpu().numpy())
            truth.append(y.cpu().numpy())
    return total_loss / seen, correct / seen, np.concatenate(probabilities), np.concatenate(truth)


def run(root):
    root = Path(root).resolve()
    report = root / 'reports/cnn'
    output = root / 'models/cnn'
    report.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if (report / 'summary.json').exists() or (output / 'best.pt').exists():
        raise FileExistsError('CNN artifacts already exist; preserve and inspect before retraining.')
    seed = TRAIN_CONFIG['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(TRAIN_CONFIG['cpu_threads'])
    # Seeds are recorded; identical results across CPU/MPS/software versions are not guaranteed.
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    images, labels = load_training_data(root, download=False)
    train_idx, val_idx = load_or_create_split(labels, root)
    subset_path = root / 'data/processed/baseline_train_indices.npy'
    subset = np.load(subset_path, allow_pickle=False)
    expected, _ = train_test_split(train_idx, train_size=12000, random_state=seed,
                                  stratify=labels[train_idx])
    assert np.array_equal(subset, expected), 'Use the identical saved baseline subset'
    assert np.all(np.bincount(labels[subset], minlength=10) == 1200)
    assert not np.intersect1d(subset, val_idx).size
    train_loader = make_loader(images, labels, subset, shuffle=True)
    val_loader = make_loader(images, labels, val_idx)
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'],
                                 weight_decay=TRAIN_CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    metadata = {'model_config': MODEL_CONFIG, 'training_config': TRAIN_CONFIG,
                'class_names': list(CLASS_NAMES), 'preprocessing': 'uint8 -> float32 / 255; (N,1,28,28); original orientation',
                'split_sha256': hashlib.sha256((root / 'data/processed/train_validation_split.npz').read_bytes()).hexdigest(),
                'subset_sha256': hashlib.sha256(subset.tobytes()).hexdigest(),
                'device': str(device), 'torch': str(torch.__version__), 'python': platform.python_version(),
                'platform': platform.platform(), 'numpy': np.__version__,
                'seed_limitations': 'Single seeded run; exact cross-device/version reproducibility is not guaranteed.'}
    (output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (report / 'architecture.txt').write_text(str(model) + '\n')
    print(f'Device: {device}; trainable parameters: {count_parameters(model):,}', flush=True)
    print(f'Training: {len(subset)}; validation: {len(val_idx)}; official test: not loaded', flush=True)
    history, best_loss, stale, best_epoch = [], float('inf'), 0, 0
    start_run = time.perf_counter()
    for epoch in range(1, TRAIN_CONFIG['max_epochs'] + 1):
        start_epoch = time.perf_counter()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        val_loss, val_accuracy, probs, truth = evaluate(model, val_loader, device)
        if not np.isfinite(val_loss):
            raise RuntimeError('Non-finite validation loss')
        improved = val_loss < best_loss - TRAIN_CONFIG['min_delta']
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            # Save CPU tensors for portable loading; include BatchNorm buffers in state_dict.
            checkpoint = {'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          'epoch': epoch, 'validation_loss': val_loss, **metadata}
            temporary = output / 'best.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(output / 'best.pt')
        else:
            stale += 1
        row = {'epoch': epoch, 'train_loss': loss_sum / seen, 'train_accuracy': correct / seen,
               'validation_loss': val_loss, 'validation_accuracy': val_accuracy,
               'validation_macro_f1': f1_score(truth, probs.argmax(1), average='macro'),
               'seconds': time.perf_counter() - start_epoch, 'checkpoint_saved': improved}
        history.append(row)
        pd.DataFrame(history).to_csv(report / 'history.csv', index=False)
        print(f"Epoch {epoch:02d}: train loss={row['train_loss']:.4f}, val loss={val_loss:.4f}, "
              f"val accuracy={val_accuracy:.4f}, val F1={row['validation_macro_f1']:.4f}, "
              f"{row['seconds']:.1f}s{' [best saved]' if improved else ''}", flush=True)
        if stale >= TRAIN_CONFIG['patience']:
            print(f'Early stopping: {stale} epochs without validation-loss improvement.', flush=True)
            break
    # Evaluate only restored best weights; last-epoch weights are not used for reported scores.
    best_model = SmallCNN().to(device)
    saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    best_model.load_state_dict(saved['state_dict'])
    final_loss, final_accuracy, probabilities, truth = evaluate(best_model, val_loader, device)
    assert np.isclose(final_loss, best_loss, rtol=1e-5, atol=1e-6)
    assert np.array_equal(truth, labels[val_idx])
    assert np.allclose(probabilities.sum(1), 1, atol=1e-5)
    assert best_epoch == int(pd.DataFrame(history).loc[lambda x: x.validation_loss.idxmin(), 'epoch'])
    pred = probabilities.argmax(1)
    np.savez_compressed(report / 'validation_predictions.npz', validation_indices=val_idx,
                        true_labels=truth, predictions=pred, probabilities=probabilities)
    pd.DataFrame({'label': np.arange(10), 'category': CLASS_NAMES,
        'validation_recall': recall_score(truth, pred, average=None, labels=np.arange(10))}).to_csv(report / 'per_class_recall.csv', index=False)
    assert not [p for p in (root / 'data').rglob('*') if p.is_file() and ('t10k' in p.name or 'fashion-mnist_test' in p.name)]
    summary = {'milestone': 4, 'status': 'complete', **metadata,
               'training_images': len(subset), 'validation_images': len(val_idx),
               'trainable_parameters': count_parameters(best_model), 'epochs_run': len(history),
               'best_epoch': best_epoch, 'early_stopped': stale >= TRAIN_CONFIG['patience'],
               'best_validation_loss': final_loss, 'validation_accuracy': accuracy_score(truth, pred),
               'validation_macro_f1': f1_score(truth, pred, average='macro'),
               'training_and_validation_seconds': time.perf_counter() - start_run,
               'checkpoint_bytes': (output / 'best.pt').stat().st_size,
               'checkpoint_reload_verified': True, 'official_test_loaded': False,
               'comparison_note': 'Same 12000 training and 12000 validation examples as baselines. CNN epoch chosen by validation loss; baselines selected by validation macro F1. Different search and optimization budgets; single seeded run. No official test scores.'}
    (report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"Best epoch {best_epoch}; validation accuracy={summary['validation_accuracy']:.4f}, "
          f"macro F1={summary['validation_macro_f1']:.4f}. Reload verified.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.project_root)

if __name__ == '__main__':
    main()
