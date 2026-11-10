import numpy as np
import pandas as pd
import pytest

from ramscore.metrics import bootstrap_ci, compute_metrics


def test_perfect_prediction():
    gt = [0, 0, 1, 2, 3, 5]
    m = compute_metrics(gt, gt)
    assert m['bacc5'] == m['bacc4'] == m['qwk'] == m['accuracy'] == 1.0
    assert m['mae'] == 0.0


def test_all_zero_prediction_is_chance_level():
    gt = [0] * 90 + [1] * 5 + [2] * 3 + [3] + [5]
    m = compute_metrics(gt, [0] * len(gt))
    assert m['accuracy'] == pytest.approx(0.90)
    assert m['bacc5'] == pytest.approx(1 / 5)
    assert m['bacc4'] == pytest.approx(1 / 4)
    assert m['binary_sensitivity'] == 0.0
    assert m['recall_ge2'] == 0.0


def test_grades_3_and_5_merge_in_bacc4():
    gt = [3, 5, 0, 1, 2]
    pred = [5, 3, 0, 1, 2]
    m = compute_metrics(gt, pred)
    assert m['bacc4'] == 1.0
    assert m['bacc5'] == pytest.approx(3 / 5)
    assert m['mae'] == pytest.approx(4 / 5)          # raw-score MAE, as in the benchmark


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    gt = rng.choice([0, 1, 2, 3, 5], size=400, p=[0.7, 0.15, 0.1, 0.03, 0.02])
    pred = np.where(rng.random(400) < 0.7, gt, 0)
    df = pd.DataFrame({'patient': np.repeat(np.arange(25), 16), 'gt': gt, 'pred': pred})
    ci = bootstrap_ci(df, n_boot=200, seed=1)
    point = compute_metrics(gt, pred)
    for k in ('bacc4', 'qwk'):
        assert ci[k][0] <= point[k] <= ci[k][1]
