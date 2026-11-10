import numpy as np
import pytest
import torch

import pandas as pd

from ramscore.compare import load_runs
from ramscore.probe import build_encoder, collapse_stats, knn_predict, linear_predict, probe


def clustered(rng, centres, n_per_class, spread=0.3):
    y = np.repeat(np.arange(len(centres)), n_per_class)
    return centres[y] + spread * rng.normal(size=(len(y), centres.shape[1])), y


def test_probes_recover_separable_classes():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(5, 16))
    x, y = clustered(rng, centres, 40)
    xt, yt = clustered(rng, centres, 40)
    assert (knn_predict(x, y, xt, 5) == yt).mean() > 0.95
    pred, c = linear_predict(x, y, x, y, xt)
    assert (pred == yt).mean() > 0.95 and c in (0.01, 0.1, 1.0)


def test_probe_returns_predictions_compare_can_load(tmp_path):
    rng = np.random.default_rng(3)
    centres = rng.normal(size=(5, 16))
    (x, y), (xt, yt) = clustered(rng, centres, 30), clustered(rng, centres, 10)
    feats, labels = {'train': x, 'val': xt, 'test': xt}, {'train': y, 'val': yt, 'test': yt}
    subsets = {'f1.0': np.arange(len(y)), 'f0.1_d0': np.arange(0, len(y), 3)}
    patients = [f'P{i % 5}' for i in range(len(yt))]
    metrics, preds = probe(feats, labels, patients, subsets, n_boot=0)
    assert set(preds) == {'knn_f1.0', 'linear_f1.0', 'knn_f0.1_d0', 'linear_f0.1_d0'}
    assert metrics['f1.0']['linear']['bacc4'] > 0.9
    key = pd.DataFrame({'case': patients, 'joint': range(len(yt)), 'patient': patients,
                        'gt': np.array([0, 1, 2, 3, 5])[yt]})
    for enc in ('a', 'b'):
        for k, pred in preds.items():
            (tmp_path / enc / k).mkdir(parents=True)
            key.assign(pred=pred).to_csv(tmp_path / enc / k / 'test_predictions.csv', index=False)
    runs, names = load_runs(['a', 'b'], str(tmp_path / '{}' / 'linear_f*'))
    assert names['a'] == ['linear_f0.1_d0', 'linear_f1.0']


def test_balanced_knn_does_not_drown_rare_class():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(0, 1, (500, 8)), rng.normal(0, 1, (10, 8)) + 0.8])
    y = np.array([0] * 500 + [1] * 10)
    query = np.full((1, 8), 0.8)
    assert knn_predict(x, y, query, 2, k=50)[0] == 1          # an unweighted vote would say 0


def test_collapse_stats_flags_collapsed_features():
    rng = np.random.default_rng(2)
    spread = collapse_stats(rng.normal(size=(500, 64)))
    collapsed = collapse_stats(np.ones((500, 64)) + 1e-4 * rng.normal(size=(500, 64)))
    assert spread['feature_sd'] == pytest.approx(spread['sd_if_uniform'], rel=0.1)
    assert collapsed['feature_sd'] < 0.01 * spread['feature_sd']
    assert spread['effective_rank'] > 40


def test_build_encoder_loads_e1_checkpoint(tmp_path):
    net, _ = build_encoder('random:resnet18')
    run = tmp_path / 'e1_x'
    run.mkdir()
    torch.save({'model': net.state_dict(), 'epoch': 24, 'backbone': 'resnet18'}, run / 'encoder_ep25.pt')
    loaded, name = build_encoder(str(run / 'encoder_ep25.pt'))
    assert name == 'e1_x_ep25'
    x = torch.randn(2, 1, 64, 64)
    assert torch.allclose(net.eval()(x), loaded.eval()(x))
