import json

import numpy as np
import pandas as pd
import pytest

from ramscore.e3 import ENCODERS, STRATEGIES, all_contrasts, collect, parse_name, run_dir, selection

GRADES = [0, 1, 2, 3, 5]
SEEDS = (42, 1337, 2026)


def test_parse_name_and_run_dir():
    assert parse_name('e3_a4_resnet34_b2_f0.1_d2_s1337') == {
        'exp': 'e3', 'A': 'a4', 'backbone': 'resnet34', 'B': 'b2', 'frac': '0.1', 'draw': 2, 'seed': 1337}
    assert parse_name('e2_a3_vits_b0_f0.1_d0_s42') is None
    assert run_dir('a1', 'b4') == 'e2/e2_a1_resnet34_b4' and run_dir('a1', 'b1') == 'e3/e3_a1_resnet34_b1'


def write_run(root, name, rng, gt, keep):
    d = root / name.split('_')[0] / name
    d.mkdir(parents=True)
    pred = np.where(rng.random(len(gt)) < keep, gt, rng.choice(GRADES, size=len(gt)))
    pd.DataFrame({'case': np.arange(len(gt)), 'patient': np.repeat([f'P{i:02d}' for i in range(30)], 20),
                  'joint': 'MCP1', 'gt': gt, 'pred': pred}).to_csv(d / 'test_predictions.csv', index=False)
    m = {k: keep for k in ('bacc4', 'bacc5', 'qwk', 'macro_f1', 'recall_ge2')}
    (d / 'metrics.json').write_text(json.dumps({'val': m, 'test': m}))


@pytest.fixture
def root(tmp_path, monkeypatch):
    """100% labels only: B4 beats B1 for DINO (a4) and ties elsewhere, so the H2 interaction is positive."""
    import ramscore.e3 as e3
    monkeypatch.setattr(e3, 'FRACTIONS', ('1.0',))
    rng = np.random.default_rng(0)
    gt = rng.choice(GRADES, size=600, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    for a in ENCODERS:
        for b in STRATEGIES:
            keep = 0.9 if (a, b) == ('a4', 'b4') else 0.3
            exp = run_dir(a, b).split('/')[0]
            for seed in SEEDS:
                write_run(tmp_path, f'{exp}_{a}_resnet34_{b}_f1.0_d0_s{seed}', rng, gt, keep)
    return tmp_path


def test_collect_reads_both_experiments(root):
    df = collect(root)
    assert len(df) == len(ENCODERS) * len(STRATEGIES) * len(SEEDS)
    assert set(df.exp) == {'e2', 'e3'} and df.loc[(df.A == 'a4') & (df.B == 'b4'), 'val_bacc4'].eq(0.9).all()
    _, best = selection(df)
    assert best == {'1.0': {'A': 'a4', 'B': 'b4'}}


def test_primary_and_interaction(root):
    df = all_contrasts(root, n_boot=300)
    primary = df[df.family == 'primary'].set_index('contrast')
    assert len(primary) == 3 and primary.bacc4_runs.eq(3).all()
    assert primary.loc['a4: b4-b1', 'bacc4_diff'] > 0.3 and primary.loc['a4: b4-b1', 'bacc4_p_holm'] < 0.05
    assert abs(primary.loc['a1: b4-b1', 'bacc4_diff']) < 0.1
    assert (primary.bacc4_p_holm >= primary.bacc4_p_hier).all()
    h2 = df[df.family == 'h2'].set_index('contrast')
    for ref in ('a1', 'a0'):           # interaction = difference of the two primary contrasts
        diff = primary.loc['a4: b4-b1', 'bacc4_diff'] - primary.loc[f'{ref}: b4-b1', 'bacc4_diff']
        assert h2.loc[f'(b4-b1) a4 - {ref}', 'bacc4_diff'] == pytest.approx(diff)
        assert h2.loc[f'(b4-b1) a4 - {ref}', 'bacc4_ci_hier'][0] > 0
    assert len(df[df.family == 'vs_b0 a4']) == 6
