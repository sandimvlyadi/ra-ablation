import json

import numpy as np
import pandas as pd
import pytest

from ramscore.e2 import all_contrasts, collect, mixed_model, parse_name, term_of

GRADES = [0, 1, 2, 3, 5]


def test_parse_name():
    assert parse_name('e2_a4_vits_b6_f0.1_d2_s1337') == {
        'A': 'a4', 'backbone': 'vits', 'B': 'b6', 'frac': '0.1', 'draw': 2, 'seed': 1337}
    assert parse_name('b0_resnet34_f1.0_s42') is None


def test_term_of():
    assert term_of('C(A, Sum)[S.a0]') == 'A'
    assert term_of('C(A, Sum)[S.a0]:C(frac, Sum)[S.0.01]') == 'A:frac'


def write_run(root, name, rng, gt, keep):
    d = root / name
    d.mkdir()
    pred = np.where(rng.random(len(gt)) < keep, gt, rng.choice(GRADES, size=len(gt)))
    pd.DataFrame({'case': np.arange(len(gt)), 'patient': np.repeat([f'P{i:02d}' for i in range(30)], 20),
                  'joint': 'MCP1', 'gt': gt, 'pred': pred}).to_csv(d / 'test_predictions.csv', index=False)
    test = {'bacc4': keep, 'bacc5': keep, 'qwk': keep, 'macro_f1': keep, 'recall_ge2': keep}
    (d / 'metrics.json').write_text(json.dumps({'test': test}))


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Three ResNet34 encoders at 10%: a4 predicts far better than a0 and a1."""
    import ramscore.e2 as e2
    monkeypatch.setattr(e2, 'ENCODERS', {'resnet34': ('a0', 'a1', 'a4')})
    monkeypatch.setattr(e2, 'FRACTIONS', ('0.1',))
    monkeypatch.setattr(e2, 'STRATEGIES', ('b0', 'b4'))
    rng = np.random.default_rng(0)
    gt = rng.choice(GRADES, size=600, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    for a, keep in (('a0', 0.2), ('a1', 0.3), ('a4', 0.8)):
        for b in ('b0', 'b4'):
            for draw in range(3):
                write_run(tmp_path, f'e2_{a}_resnet34_{b}_f0.1_d{draw}_s42', rng, gt, keep)
    return tmp_path


def test_primary_contrasts_and_holm(root):
    df = all_contrasts(root, n_boot=300)
    primary = df[df.primary]
    assert len(primary) == 4                                   # a4 vs a0 and a1, B0 and B4
    assert (primary.bacc4_diff > 0.2).all() and (primary.p_holm < 0.05).all()
    assert (primary.p_holm >= primary.bacc4_p_hier).all()
    assert primary.n_runs.eq(3).all()


def test_collect_reads_factors(root):
    df = collect(root)
    assert len(df) == 18 and set(df.encoder) == {'a0_resnet34', 'a1_resnet34', 'a4_resnet34'}
    assert df.loc[df.A == 'a4', 'bacc4'].eq(0.8).all()


def test_mixed_model_finds_encoder_effect():
    rng = np.random.default_rng(1)
    rows = []
    for a, base in (('a0', 0.25), ('a1', 0.30), ('a4', 0.35)):
        for b, shift in (('b0', 0.0), ('b4', 0.02), ('b6', 0.03)):
            for frac in ('0.1', '1.0'):
                for draw in range(3):
                    for seed in (42, 1337):
                        rows.append({'A': a, 'B': b, 'frac': frac, 'draw': draw, 'seed': seed, 'backbone': 'resnet34',
                                     'bacc4': base + shift + 0.1 * (frac == '1.0') + rng.normal(0, 0.01)})
    res = mixed_model(pd.DataFrame(rows), 'resnet34')
    tests = {t['term']: t for t in res['tests']}
    assert tests['A']['df'] == 2 and tests['A']['p'] < 1e-6
    assert tests['A:fraction']['p'] > 0.001                  # no interaction was simulated
