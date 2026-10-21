"""v2 experiment: controlled ablations of training data, augmentation and a deeper CNN.

Run from the project root (inside Docker):
  python -m src.train_v2 train --config deep_48k_aug --seed 42   # one ablation run
  python -m src.train_v2 report                                  # validation comparison table
  python -m src.train_v2 select                                  # freeze the choice (validation only)
  python -m src.train_v2 test                                    # official test, exactly once

Every configuration shares the optimizer, early stopping, validation split and seed handling;
each differs from its neighbour in one factor. Outputs go to runs/v2/ only; v1 code, models,
reports and data are read, never written. The official test set is opened only by `test`,
after `select` has frozen the choice, and training is refused from then on.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from .data import CLASS_NAMES, TRAIN_FILES, load_training_data, load_or_create_split, scale_pixels
from .model import SmallCNN, count_parameters

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'runs/v2'
COMMON = {'batch_size': 128, 'optimizer': 'Adam', 'learning_rate': 0.001, 'weight_decay': 0.0001,
          'max_epochs': 40, 'patience': 5, 'selection_metric': 'minimum validation cross-entropy loss'}
CONFIGS = {
    'v1_12k':       {'arch': 'v1', 'budget': 12000, 'augment': False},  # v1 recipe, reference
    'v1_48k':       {'arch': 'v1', 'budget': 48000, 'augment': False},  # + all training data
    'v1_48k_aug':   {'arch': 'v1', 'budget': 48000, 'augment': True},   # + augmentation
    'deep_48k':     {'arch': 'deep', 'budget': 48000, 'augment': False},  # + deeper CNN
    'deep_48k_aug': {'arch': 'deep', 'budget': 48000, 'augment': True},   # all three
}
AUGMENTATION = 'random horizontal flip p=0.5; random translation of up to 2 px in each axis, zero fill'
BOOTSTRAP = {'repeats': 2000, 'seed': 42, 'interval': 0.95}
SELECTION_RULE = (
    'Validation only. Each seed run is scored per image on the shared 12,000 validation images. An accuracy '
    'difference gets a 95% interval from a paired two-level bootstrap: validation images are resampled jointly '
    'for both configurations (paired), seed runs independently within each configuration. "Best" is the highest '
    'mean accuracy. The candidate is the simplest configuration (order: ' + ' < '.join(CONFIGS) + ') whose '
    'best-minus-candidate interval includes zero. It is adopted only if its interval against v1_12k lies '
    'entirely above zero; otherwise v1 is retained and no v2 test evaluation runs. The seed-42 checkpoint of '
    'the adopted configuration is the v2 model; seeds are not cherry-picked.')
RULE_HISTORY = (
    'Revised 2026-10-21, before any selection ran. It replaced an unpaired two-standard-error test of the best '
    'mean against v1_12k, which ignored the shared validation images, used too few degrees of freedom and never '
    'compared the deeper CNN or augmentation with v1_48k. Seed-42 validation results for v1_12k (complete) and '
    'v1_48k, v1_48k_aug and deep_48k (in progress) were visible when the rule was revised.')
EFFECTS = {'more training data': ('v1_48k', 'v1_12k'),
           'augmentation (v1 CNN)': ('v1_48k_aug', 'v1_48k'),
           'deeper CNN (no augmentation)': ('deep_48k', 'v1_48k'),
           'augmentation (deep CNN)': ('deep_48k_aug', 'deep_48k'),
           'deeper CNN (with augmentation)': ('deep_48k_aug', 'v1_48k_aug')}


class DeepCNN(nn.Module):
    """Three VGG-style blocks (two 3x3 convolutions each) with the v1 regularizers kept."""
    def __init__(self):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                    nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Dropout2d(0.1)]
        self.features = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 128))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 3 * 3, 256), nn.ReLU(),
                                        nn.Dropout(0.4), nn.Linear(256, 10))

    def forward(self, images):
        return self.classifier(self.features(images))


def build(arch):
    return SmallCNN() if arch == 'v1' else DeepCNN()


def augment(x, generator):
    """Label-preserving for upright, centred items: no vertical flips, rotations or crops."""
    n = len(x)
    flip = torch.rand(n, generator=generator) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    shifts = torch.randint(-2, 3, (n, 2), generator=generator)
    padded = F.pad(x, (2, 2, 2, 2))
    grid = torch.arange(28)
    rows = (2 + shifts[:, 0, None] + grid)[:, :, None]
    cols = (2 + shifts[:, 1, None] + grid)[:, None, :]
    return padded[torch.arange(n)[:, None, None], 0, rows, cols][:, None]


def batches(x, y, batch_size, generator=None):
    order = torch.randperm(len(x), generator=generator) if generator is not None else torch.arange(len(x))
    for start in range(0, len(x), batch_size):
        index = order[start:start + batch_size]
        yield x[index], y[index]


def evaluate(model, x, y):
    model.eval()
    loss, correct, probabilities = 0.0, 0, []
    with torch.inference_mode():
        for xb, yb in batches(x, y, 512):
            logits = model(xb)
            loss += F.cross_entropy(logits, yb, reduction='sum').item()
            correct += (logits.argmax(1) == yb).sum().item()
            probabilities.append(logits.softmax(1))
    return loss / len(x), correct / len(x), torch.cat(probabilities).numpy()


def tensors(images, labels, indices):
    return (torch.from_numpy(scale_pixels(images[indices])[:, None]),
            torch.from_numpy(labels[indices].astype(np.int64)))


def prepare_root():
    """Training archives only; copied from a previous run when available, else downloaded."""
    if (V2 / 'selection.json').exists():
        raise FileExistsError('v2 choice is frozen and the test set may be open; no further training.')
    raw = V2 / 'data/raw/FashionMNIST'
    raw.mkdir(parents=True, exist_ok=True)
    for name in TRAIN_FILES:
        cached = ROOT / 'runs/reproduce/data/raw/FashionMNIST' / name
        if not (raw / name).exists() and cached.exists():
            shutil.copy2(cached, raw / name)
    images, labels = load_training_data(V2, download=True)
    assert not [p for p in (V2 / 'data').rglob('*t10k*')], 'Test archives must not exist during training'
    train_idx, val_idx = load_or_create_split(labels, V2)
    # The identical 12,000-image subset v1 trained on, also used to measure the train-val gap.
    subset, _ = train_test_split(train_idx, train_size=12000, random_state=42, stratify=labels[train_idx])
    return images, labels, train_idx, val_idx, subset


def train(name, seed, threads):
    config = CONFIGS[name]
    out = V2 / 'experiments' / f'{name}-s{seed}'
    if (out / 'summary.json').exists():
        print(f'{out.name} already complete; skipping.', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    images, labels, train_idx, val_idx, subset = prepare_root()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    fit_idx = subset if config['budget'] == 12000 else train_idx
    x_fit, y_fit = tensors(images, labels, fit_idx)
    x_val, y_val = tensors(images, labels, val_idx)
    x_gap, y_gap = tensors(images, labels, subset)
    model = build(config['arch'])
    optimizer = torch.optim.Adam(model.parameters(), lr=COMMON['learning_rate'],
                                 weight_decay=COMMON['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    print(f"{out.name}: {count_parameters(model):,} parameters; {len(fit_idx)} training images; "
          f"augmentation={'on' if config['augment'] else 'off'}", flush=True)
    history, best_loss, best_epoch, stale, start = [], float('inf'), 0, 0, time.perf_counter()
    for epoch in range(1, COMMON['max_epochs'] + 1):
        tick = time.perf_counter()
        model.train()
        loss_sum = 0.0
        for xb, yb in batches(x_fit, y_fit, COMMON['batch_size'], generator):
            if config['augment']:
                xb = augment(xb, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
        val_loss, val_acc, _ = evaluate(model, x_val, y_val)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({'state_dict': {k: v.detach().clone() for k, v in model.state_dict().items()},
                        'epoch': epoch, 'config': name, 'arch': config['arch'], 'seed': seed,
                        'class_names': list(CLASS_NAMES)}, out / 'best.pt')
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_loss': loss_sum / len(fit_idx), 'validation_loss': val_loss,
                        'validation_accuracy': val_acc, 'seconds': time.perf_counter() - tick,
                        'checkpoint_saved': improved})
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print(f"  epoch {epoch:02d}: train loss={history[-1]['train_loss']:.4f} val loss={val_loss:.4f} "
              f"val acc={val_acc:.4f} {history[-1]['seconds']:.0f}s{' *' if improved else ''}", flush=True)
        if stale >= COMMON['patience']:
            break
    model.load_state_dict(torch.load(out / 'best.pt', weights_only=True)['state_dict'])
    val_loss, val_acc, probabilities = evaluate(model, x_val, y_val)
    # Clean (eval-mode, unaugmented) scores on the shared 12k training subset: a like-for-like gap.
    train_loss, train_acc, _ = evaluate(model, x_gap, y_gap)
    pred = probabilities.argmax(1)
    truth = y_val.numpy()
    summary = {'config': name, 'seed': seed, **config, **COMMON,
               'augmentation': AUGMENTATION if config['augment'] else 'none',
               'parameters': count_parameters(model), 'training_images': len(fit_idx),
               'validation_images': len(val_idx), 'epochs_run': len(history), 'best_epoch': best_epoch,
               'validation_loss': val_loss, 'validation_accuracy': val_acc,
               'validation_macro_f1': f1_score(truth, pred, average='macro'),
               'shirt_validation_recall': recall_score(truth, pred, labels=[6], average=None)[0],
               'gap_set': 'v1 12k training subset, eval mode, no augmentation',
               'train_loss': train_loss, 'train_accuracy': train_acc,
               'accuracy_gap': train_acc - val_acc, 'loss_gap': val_loss - train_loss,
               'seconds': time.perf_counter() - start, 'torch': torch.__version__,
               'official_test_loaded': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"{out.name}: best epoch {best_epoch}; val acc={val_acc:.4f}, val loss={val_loss:.4f}, "
          f"train-val acc gap={train_acc - val_acc:+.4f}", flush=True)


def collect():
    rows = [json.loads(p.read_text()) for p in sorted((V2 / 'experiments').glob('*/summary.json'))]
    if not rows:
        raise FileNotFoundError('No completed v2 runs.')
    return pd.DataFrame(rows)


def report():
    runs = collect()
    runs.to_csv(V2 / 'ablation_runs.csv', index=False)
    metrics = ['validation_accuracy', 'validation_loss', 'validation_macro_f1', 'shirt_validation_recall',
               'train_accuracy', 'accuracy_gap', 'loss_gap', 'best_epoch', 'seconds']
    table = runs.groupby('config')[metrics].agg(['mean', 'std', 'count']).reindex(list(CONFIGS)).dropna(how='all')
    table.to_csv(V2 / 'ablation_summary.csv')
    with pd.option_context('display.width', 200, 'display.max_columns', 50, 'display.float_format', '{:.4f}'.format):
        print(runs.groupby('config')[metrics].mean().reindex(list(CONFIGS)).dropna(how='all'))
        print('\nStandard deviation across seeds:')
        print(runs.groupby('config')[['validation_accuracy', 'validation_loss']].std()
              .reindex(list(CONFIGS)).dropna(how='all'))
    return runs


def validation_correctness(runs, images, labels, val_idx):
    """Per-image validation correctness for every finished run, as {config: (seeds, images) bool}.

    Predictions are re-inferred from each best checkpoint (training did not store them), checked against
    the accuracy recorded during training, and cached next to the checkpoint.
    """
    x_val, y_val = tensors(images, labels, val_idx)
    correct = {}
    for row in runs.sort_values(['config', 'seed']).itertuples():
        out = V2 / 'experiments' / f'{row.config}-s{row.seed}'
        cache = out / 'validation_predictions.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                assert np.array_equal(saved['validation_indices'], val_idx), cache
                pred = saved['predictions']
        else:
            checkpoint = torch.load(out / 'best.pt', weights_only=True)
            model = build(checkpoint['arch'])
            model.load_state_dict(checkpoint['state_dict'])
            _, accuracy, probabilities = evaluate(model, x_val, y_val)
            assert abs(accuracy - row.validation_accuracy) < 1e-9, f'{out.name}: reloaded accuracy differs'
            pred = probabilities.argmax(1)
            np.savez_compressed(cache, validation_indices=val_idx, predictions=pred,
                                probabilities=probabilities.astype(np.float32))
        correct.setdefault(row.config, []).append(pred == labels[val_idx])
    return {config: np.stack(rows) for config, rows in correct.items()}


def difference_interval(a, b, rng, repeats=BOOTSTRAP['repeats']):
    """Accuracy(a) - accuracy(b) with a paired two-level bootstrap interval.

    a and b are (seeds, images) correctness arrays over the same validation images. Images are resampled
    jointly (the pairing); seed runs are resampled independently per configuration.
    """
    n = a.shape[1]
    draws = np.empty(repeats)
    for r in range(repeats):
        weights = np.bincount(rng.integers(0, n, n), minlength=n)
        mean_a = a[rng.integers(0, len(a), len(a))].mean(0)
        mean_b = b[rng.integers(0, len(b), len(b))].mean(0)
        draws[r] = weights @ (mean_a - mean_b) / n
    tail = (1 - BOOTSTRAP['interval']) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return {'difference': float(a.mean() - b.mean()), 'low': float(low), 'high': float(high)}


def apply_rule(correct, rng):
    """SELECTION_RULE on {config: (seeds, images) correctness}; pure, so it can be tested directly."""
    order = list(CONFIGS)
    means = {c: float(correct[c].mean()) for c in order}
    best = max(order, key=lambda c: (means[c], -order.index(c)))
    versus_best = {c: difference_interval(correct[best], correct[c], rng) for c in order if c != best}
    candidate = next(c for c in order if c == best or versus_best[c]['low'] <= 0)
    versus_v1 = (difference_interval(correct[candidate], correct['v1_12k'], rng)
                 if candidate != 'v1_12k' else None)
    effects = {name: {'comparison': f'{a} - {b}', **difference_interval(correct[a], correct[b], rng)}
               for name, (a, b) in EFFECTS.items()}
    return {'validation_accuracy': means, 'best': best, 'best_minus_config': versus_best,
            'candidate': candidate, 'candidate_minus_v1_12k': versus_v1,
            'adopted': bool(versus_v1 is not None and versus_v1['low'] > 0), 'effects': effects}


def select():
    """Apply the declared rule to validation results and freeze it before any test access."""
    path = V2 / 'selection.json'
    if path.exists():
        raise FileExistsError('Selection already frozen.')
    runs = report()
    seeds = runs.groupby('config')['seed'].apply(frozenset)
    if set(seeds.index) != set(CONFIGS) or seeds.nunique() != 1 or len(seeds.iloc[0]) < 2:
        raise ValueError('Select needs the same seeds (at least two) for all five configurations.')
    images, labels, _, val_idx, _ = prepare_root()
    outcome = apply_rule(validation_correctness(runs, images, labels, val_idx),
                         np.random.default_rng(BOOTSTRAP['seed']))
    chosen = outcome['candidate'] if outcome['adopted'] else 'v1_12k'
    checkpoint = V2 / 'experiments' / f'{chosen}-s42/best.pt'
    selection = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'rule': SELECTION_RULE,
                 'rule_history': RULE_HISTORY, 'bootstrap': BOOTSTRAP, 'seeds': sorted(seeds.iloc[0]),
                 **outcome, 'chosen_config': chosen, 'v2_adopted': outcome['adopted'],
                 'checkpoint': str(checkpoint.relative_to(V2)) if outcome['adopted'] else None,
                 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest() if outcome['adopted'] else None,
                 'official_test_loaded': False}
    path.write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps({k: selection[k] for k in ('best', 'candidate', 'candidate_minus_v1_12k',
                                                 'v2_adopted', 'chosen_config', 'effects')}, indent=2))


def test():
    """Evaluate the frozen v2 checkpoint on the official test set, exactly once."""
    from .evaluate import load_official_test
    selection = json.loads((V2 / 'selection.json').read_text())
    if not selection['v2_adopted']:
        raise SystemExit('Selection retained v1; there is no v2 model to evaluate on the test set.')
    out = V2 / 'test'
    started = out / 'started.json'
    if started.exists():
        raise FileExistsError('v2 test evaluation was already started; it runs once and is never repeated.')
    checkpoint_path = V2 / selection['checkpoint']
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == selection['checkpoint_sha256']
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model = build(checkpoint['arch'])
    model.load_state_dict(checkpoint['state_dict'])
    out.mkdir(parents=True, exist_ok=True)
    # Marked before the test set is opened, so a crash can't be followed by a quiet second attempt.
    started.write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(),
                                   'checkpoint_sha256': selection['checkpoint_sha256']}) + '\n')
    images, labels = load_official_test(V2)
    x, y = torch.from_numpy(scale_pixels(images)[:, None]), torch.from_numpy(labels.astype(np.int64))
    loss, accuracy, probabilities = evaluate(model, x, y)
    pred = probabilities.argmax(1)
    v1 = pd.read_csv(ROOT / 'reports/final_evaluation/test_results.csv').set_index('model').loc['cnn']
    with np.load(ROOT / 'reports/final_evaluation/test_predictions.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['test_indices'], np.arange(len(labels)))
        v1_pred = saved['cnn']
    recall = recall_score(labels, pred, average=None, labels=np.arange(10))
    summary = {'chosen_config': selection['chosen_config'], 'test_accuracy': accuracy_score(labels, pred),
               'test_macro_f1': f1_score(labels, pred, average='macro'), 'test_loss': loss,
               'v1_test_accuracy': float(v1['test_accuracy']), 'v1_test_macro_f1': float(v1['test_macro_f1']),
               'v2_correct_v1_wrong': int(((pred == labels) & (v1_pred != labels)).sum()),
               'v1_correct_v2_wrong': int(((pred != labels) & (v1_pred == labels)).sum()),
               'per_class_recall': dict(zip(CLASS_NAMES, recall.tolist())),
               'note': 'Single test evaluation of a validation-selected model; not used for any tuning.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('train')
    run.add_argument('--config', choices=CONFIGS, required=True)
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--threads', type=int, default=2)
    for name in ('report', 'select', 'test'):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == 'train':
        train(args.config, args.seed, args.threads)
    else:
        {'report': report, 'select': select, 'test': test}[args.command]()


if __name__ == '__main__':
    main()
