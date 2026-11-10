import json

import numpy as np
import pandas as pd
import pytest

from ramscore import rq0

DESIGNS = {'1.0': [(0, 42), (0, 1337), (0, 2026)], '0.1': [(0, 42), (1, 42), (2, 42)]}
JOINTS = ['MCP-T', 'MCP-I', 'MCP-M', 'MCP-R']


def write_run(folder, shift, seed, n=120):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    gt = np.tile([0, 0, 0, 1, 1, 2, 3, 5], n // 8)
    pred = np.where(rng.random(len(gt)) < 0.5 + shift, gt, 0)
    pd.DataFrame({'case': [f'c{i // 4}' for i in range(len(gt))], 'patient': [f'P{i % 12}' for i in range(len(gt))],
                  'joint': [JOINTS[i % 4] for i in range(len(gt))], 'gt': gt, 'pred': pred}
                 ).to_csv(folder / 'test_predictions.csv', index=False)
    (folder / 'metrics.json').write_text(json.dumps({'best_epoch': 10, 'val': {'bacc4': 0.4 + shift, 'qwk': 0.3}}))


@pytest.fixture
def runs(tmp_path):
    new, old = tmp_path / 'runs', tmp_path / 'old_e8'
    for frac, tags in DESIGNS.items():
        for d, s in tags:
            write_run(new / 'rq0' / f'rq0-01_resnet34_b2_f{frac}_d{d}_s{s}', 0.1, s + d)
            write_run(new / 'rq0' / f'rq0-02_convnext_tiny_b2_f{frac}_d{d}_s{s}', 0.2, s + d + 7)
            write_run(old / f'e8_base_a1_resnet34_b2_f{frac}_d{d}_s{s}', 0.1, s + d)   # same stream = identical run
    return new, old


def test_primary_family_has_two_contrasts_with_holm(runs):
    res = rq0.analyse(*runs, n_boot=100)
    assert [r['frac'] for r in res['primary']] == ['1.0', '0.1']
    assert all('p_holm' in r['bacc4'] and r['runs'] == 3 and 'verdict' in r for r in res['primary'])
    assert len(res['supporting']) == 2 and all('p_holm' not in r['bacc4'] for r in res['supporting'])
    assert len(res['predictions']) == 4 and len(res['summaries']) == 4


def test_identical_rerun_passes_the_reproduction_check(runs):
    rows = rq0.analyse(*runs, n_boot=50)['reproduction']
    assert len(rows) == 6
    assert all(r['identical'] == 1.0 and r['diff_bacc4'] == 0.0 and not r['flag'] for r in rows)


def test_reproduction_flags_a_drifted_run(runs):
    new, old = runs
    path = new / 'rq0' / 'rq0-01_resnet34_b2_f1.0_d0_s42' / 'test_predictions.csv'
    df = pd.read_csv(path)
    df['pred'] = 0
    df.to_csv(path, index=False)
    rows = {r['tag']: r for r in rq0.reproduction(rq0.arm_glob(new, 'rq0-01', '1.0'), rq0.e8_glob(old, '1.0'))}
    assert rows['d0_s42']['flag'] and rows['d0_s42']['identical'] < 1.0
    assert not rows['d0_s1337']['flag']


def test_reproduction_refuses_other_test_joints(runs):
    new, old = runs
    path = old / 'e8_base_a1_resnet34_b2_f0.1_d1_s42' / 'test_predictions.csv'
    df = pd.read_csv(path)
    df.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError):
        rq0.reproduction(rq0.arm_glob(new, 'rq0-01', '0.1'), rq0.e8_glob(old, '0.1'))


def test_analysis_waits_for_the_runs(tmp_path):
    res = rq0.analyse(tmp_path, e8_root='', n_boot=50)
    assert all(v == [] for v in res.values())


@pytest.mark.parametrize('diff, p, expected', [(-0.06, 0.01, 'bermakna'), (0.05, 0.04, 'bermakna'),
                                               (-0.03, 0.01, 'kecil'), (-0.09, 0.05, 'tidak terbukti')])
def test_verdict_applies_the_locked_rules(diff, p, expected):
    assert rq0.verdict({'diff': diff, 'p_holm': p}) == expected


def test_report_names_methods_and_states_post_hoc(runs):
    text = rq0.report(rq0.analyse(*runs, n_boot=100))
    assert '*Post hoc*' in text
    assert 'ConvNeXt-Tiny' in text and 'ResNet34 (rujukan)' in text and 'Uji reproduksi' in text
