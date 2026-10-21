"""v2 augmentation, architecture, selection rule and test-set guards (no training, no test data)."""
import json

import numpy as np
import pytest
import torch

from src import train_v2


def test_augment_only_flips_and_shifts():
    x = torch.rand(64, 1, 28, 28)
    out = train_v2.augment(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    for original, moved in zip(x, out):
        candidates = [torch.roll(torch.nn.functional.pad(img, (2, 2, 2, 2)), (dy, dx), (1, 2))[:, 2:30, 2:30]
                      for img in (original, original.flip(-1)) for dy in range(-2, 3) for dx in range(-2, 3)]
        assert any(torch.equal(moved, c) for c in candidates)


def test_deep_cnn_shape():
    model = train_v2.DeepCNN().eval()
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


def correctness(accuracies, images=3000, seed=0):
    rng = np.random.default_rng(seed)
    return {c: rng.random((3, images)) < accuracies[c] for c in train_v2.CONFIGS}


def test_rule_prefers_simplest_config_within_noise():
    acc = {'v1_12k': 0.88, 'v1_48k': 0.920, 'v1_48k_aug': 0.921, 'deep_48k': 0.922, 'deep_48k_aug': 0.923}
    outcome = train_v2.apply_rule(correctness(acc), np.random.default_rng(42))
    assert outcome['candidate'] == 'v1_48k' and outcome['adopted']


def test_rule_picks_clear_winner():
    acc = {'v1_12k': 0.88, 'v1_48k': 0.90, 'v1_48k_aug': 0.90, 'deep_48k': 0.90, 'deep_48k_aug': 0.95}
    outcome = train_v2.apply_rule(correctness(acc), np.random.default_rng(42))
    assert outcome['best'] == outcome['candidate'] == 'deep_48k_aug' and outcome['adopted']


def test_rule_keeps_v1_without_real_improvement():
    acc = dict.fromkeys(train_v2.CONFIGS, 0.9)
    outcome = train_v2.apply_rule(correctness(acc), np.random.default_rng(42))
    assert not outcome['adopted']


def test_interval_is_paired_over_images():
    base = np.tile(np.random.default_rng(1).random(3000) < 0.9, (3, 1))
    base[:, :15] = False
    better = base.copy()
    better[:, :15] = True  # +0.5 points on the same images; an unpaired test (SE ~0.8 points) would miss it
    interval = train_v2.difference_interval(better, base, np.random.default_rng(0), repeats=500)
    assert interval['difference'] == pytest.approx(0.005) and interval['low'] > 0


def test_training_refused_after_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(train_v2, 'V2', tmp_path)
    (tmp_path / 'selection.json').write_text('{}')
    with pytest.raises(FileExistsError):
        train_v2.prepare_root()


def test_test_set_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(train_v2, 'V2', tmp_path)
    (tmp_path / 'selection.json').write_text(json.dumps({'v2_adopted': False}))
    with pytest.raises(SystemExit):
        train_v2.test()
    (tmp_path / 'selection.json').write_text(json.dumps({'v2_adopted': True}))
    (tmp_path / 'test').mkdir()
    (tmp_path / 'test/started.json').write_text('{}')
    with pytest.raises(FileExistsError):
        train_v2.test()
