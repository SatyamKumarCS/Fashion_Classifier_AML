"""Committed results and hash-protected files are unchanged; installed models are accounted for."""
import json

import pandas as pd

from conftest import load_script


def test_protected_artifacts_verify(root):
    report = load_script('verify_artifacts').check(root)
    assert report['failures'] == [], report


def test_committed_benchmark_is_unchanged(root):
    results = pd.read_csv(root / 'reports/final_evaluation/test_results.csv').set_index('model')
    assert results.loc['cnn', 'test_accuracy'] == 0.89
    assert results.loc['pixels_random_forest', 'test_accuracy'] == 0.8527
    frozen = json.loads((root / 'reports/final_evaluation/frozen-models.json').read_text())
    assert frozen['deployment_choice_before_test'] == 'cnn'
