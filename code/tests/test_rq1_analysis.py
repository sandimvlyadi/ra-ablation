import json

import numpy as np
import pandas as pd
import pytest

from ramscore import rq1

DESIGNS = {'1.0': [(0, 42), (0, 1337), (0, 2026)], '0.1': [(0, 42), (1, 42), (2, 42)]}
JOINTS = ['MCP-T', 'MCP-I', 'MCP-M', 'MCP-R']
NAMES = {'rq0-01': 'resnet34', 'rq1-01': 'resnet34_scratch', 'rq1-02': 'resnet34_dino', 'rq1-03': 'resnet34_dinochrono',
         'rq1-04': 'vits_imagenet', 'rq1-05': 'vits_mae', 'rq1-06': 'vits_dino', 'rq1-17a': 'resnet34_imagenet_lp',
         'rq1-17b': 'resnet34_imagenet_lpft'}


def write_run(folder, shift, seed, val, n=120, qwk=0.3):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    gt = np.tile([0, 0, 0, 1, 1, 2, 3, 5], n // 8)
    pred = np.where(rng.random(len(gt)) < 0.5 + shift, gt, 0)
    pd.DataFrame({'case': [f'c{i // 4}' for i in range(len(gt))], 'patient': [f'P{i % 12}' for i in range(len(gt))],
                  'joint': [JOINTS[i % 4] for i in range(len(gt))], 'gt': gt, 'pred': pred}
                 ).to_csv(folder / 'test_predictions.csv', index=False)
    (folder / 'metrics.json').write_text(json.dumps({'best_epoch': 10, 'val': {'bacc4': val, 'qwk': qwk},
                                                     'test': {'bacc4': val - 0.01}}))


def make_arm(root, arm, shift=0.1, val=0.4, fracs=('1.0', '0.1'), qwk=0.3):
    sub = 'rq0' if arm.startswith('rq0') else 'rq1'
    for frac in fracs:
        for d, s in DESIGNS[frac]:
            write_run(root / sub / f'{arm}_{NAMES.get(arm, "resnet34_x")}_b2_f{frac}_d{d}_s{s}', shift, s + d, val, qwk=qwk)


@pytest.fixture
def runs(tmp_path):
    vals = {'rq0-01': 0.40, 'rq1-01': 0.30, 'rq1-02': 0.42, 'rq1-03': 0.41, 'rq1-04': 0.39, 'rq1-05': 0.45,
            'rq1-06': 0.38, 'rq1-17a': 0.35, 'rq1-17b': 0.41}
    for arm, v in vals.items():
        make_arm(tmp_path, arm, shift=v - 0.3, val=v)
    return tmp_path


def test_family_sizes_are_fixed_and_missing_arms_enter_holm_with_p_one(runs):
    res = rq1.analyse(runs, n_boot=50)
    assert {k: len(v) for k, v in res['families'].items()} == {'F1': 16, 'F2': 14, 'F3': 8, 'F4': 10}
    f1 = res['families']['F1']
    done = [r for r in f1 if r['status'] == 'dihitung']
    assert {r['arm'] for r in done} == {'rq1-01', 'rq1-02'} and len(done) == 4
    missing = [r for r in f1 if r['status'] != 'dihitung']
    assert all(r['bacc4']['p'] == 1.0 and r['bacc4']['p_holm'] == 1.0 for r in missing)
    assert all(r['verdict'] in ('bermakna', 'kecil', 'tidak terbukti') for r in done)
    assert all(r['runs'] == 3 for r in done)


def test_holm_uses_the_full_family_size(runs):
    f1 = rq1.analyse(runs, n_boot=50)['families']['F1']
    raw = sorted(r['bacc4']['p'] for r in f1 if r['status'] == 'dihitung')
    adjusted = sorted(r['bacc4']['p_holm'] for r in f1 if r['status'] == 'dihitung')
    assert adjusted[0] == pytest.approx(min(1.0, raw[0] * 16))


def test_selections_are_provisional_while_candidates_are_missing(runs):
    res = rq1.analyse(runs, n_boot=50)
    best = res['selections']['best_existing_1.0']
    assert best['provisional'] and 'rq1-07' in best['missing'] and best['arm'] == 'rq1-05'
    ssl = res['selections']['best_ssl_1.0']
    assert ssl['provisional'] and set(ssl['missing']) == {'rq1-08', 'rq1-10'} and ssl['arm'] == 'rq1-05'
    dependent = [r for r in res['families']['F3'] if r['base_key'] == rq1.BEST]
    assert all(r['status'] == 'tidak dijalankan' and 'belum final' in r['reason'] for r in dependent)


def test_selection_uses_validation_only_and_breaks_ties_by_candidate_order(tmp_path):
    make_arm(tmp_path, 'rq1-02', shift=0.0, val=0.45)
    make_arm(tmp_path, 'rq1-03', shift=0.3, val=0.45)   # much better on test, same validation
    s = rq1.select(tmp_path, ('rq1-02', 'rq1-03'), ('1.0', '0.1'))
    assert s['arm'] == 'rq1-02' and not s['provisional']


def test_final_selection_resolves_the_dependent_contrast(runs):
    for arm in ('rq1-08', 'rq1-10'):
        make_arm(runs, arm, shift=0.05, val=0.30)
    make_arm(runs, 'rq1-17c', shift=0.2, val=0.4)
    res = rq1.analyse(runs, n_boot=50)
    assert not res['selections']['best_ssl_1.0']['provisional']
    rows = [r for r in res['families']['F4'] if r['arm'] == 'rq1-17c']
    assert all(r['status'] == 'dihitung' and r['base'] == 'rq1-05' for r in rows)


def test_summary_reports_test_minus_validation(runs):
    rows = {(r['frac'], r['arm']): r for r in rq1.analyse(runs, n_boot=20)['summaries']}
    assert rows[('1.0', 'rq1-02')]['test_minus_val'] == pytest.approx(-0.01)
    assert rows[('0.1', 'rq0-01')]['runs'] == 3


def test_report_renders(runs):
    text = rq1.report(rq1.analyse(runs, n_boot=20))
    assert 'Post hoc' in text and 'Holm atas 16 kontras' in text and 'tidak dijalankan' in text


def test_qwk_gate_excludes_a_high_bacc4_arm_with_low_qwk(tmp_path):
    make_arm(tmp_path, 'rq0-01', val=0.40, qwk=0.24)
    make_arm(tmp_path, 'rq1-01', val=0.45, qwk=0.05)     # wins on BACC4, not ordinal
    make_arm(tmp_path, 'rq1-02', val=0.42, qwk=0.2401)   # equal to the floor at 3 decimals: passes
    cands = ('rq0-01', 'rq1-01', 'rq1-02')
    assert rq1.select(tmp_path, cands, ('0.1',))['arm'] == 'rq1-01'
    gated = rq1.select(tmp_path, cands, ('0.1',), qwk_floor_arm='rq0-01')
    assert gated['arm'] == 'rq1-02' and gated['excluded'] == ['rq1-01'] and gated['qwk_floor'] == pytest.approx(0.24)
    make_arm(tmp_path / 'b', 'rq1-01', val=0.45, qwk=0.05)   # floor arm missing: no selection
    assert rq1.select(tmp_path / 'b', cands, ('0.1',), qwk_floor_arm='rq0-01')['provisional']


def finish_best_candidates(root, skip=()):
    for arm in rq1.BEST_CANDIDATES:
        if arm not in skip and not list(root.glob(f'rq*/{arm}_*')):
            make_arm(root, arm, shift=0.0, val=0.30)


def test_sensitivity_rows_raw_p_only_and_families_unchanged(runs):
    finish_best_candidates(runs)
    make_arm(runs, 'rq1-13', shift=0.2, val=0.4)
    res = rq1.analyse(runs, n_boot=50)
    assert {k: len(v) for k, v in res['families'].items()} == {'F1': 16, 'F2': 14, 'F3': 8, 'F4': 10}
    sens = res['sensitivity']
    assert len(sens) == 6 and {r['family'] for r in sens} == {'F2', 'F3'}
    assert all('p_holm' not in r.get('bacc4', {}) and 'verdict' not in r for r in sens)
    rows = {(r['arm'], r['frac']): r for r in sens}
    assert res['selections']['best_existing_qwk_1.0']['arm'] == 'rq1-05'
    assert rows[('rq1-13', '1.0')]['status'] == 'sama dengan analisis utama'


def test_sensitivity_differs_from_primary_when_the_winner_fails_the_gate(tmp_path):
    make_arm(tmp_path, 'rq0-01', val=0.40, qwk=0.24)
    make_arm(tmp_path, 'rq1-01', val=0.50, qwk=0.05)
    make_arm(tmp_path, 'rq1-02', val=0.42, qwk=0.30)
    finish_best_candidates(tmp_path)
    make_arm(tmp_path, 'rq1-13', shift=0.2, val=0.4)
    res = rq1.analyse(tmp_path, n_boot=50)
    primary = [r for r in res['families']['F3'] if r['arm'] == 'rq1-13' and r['base_key'] == rq1.BEST]
    assert all(r['base'] == 'rq1-01' and r['status'] == 'dihitung' for r in primary)
    sens = [r for r in res['sensitivity'] if r['arm'] == 'rq1-13']
    assert all(r['base'] == 'rq1-02' and r['status'] == 'dihitung' and 0 <= r['bacc4']['p'] <= 1 for r in sens)
    text = rq1.report(res)
    assert 'Analisis sensitivitas' in text and 'tidak lolos gate: Scratch ResNet34' in text
