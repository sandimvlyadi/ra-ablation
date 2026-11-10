import numpy as np
import pandas as pd
import pytest

from ramscore.compare import (RunPicks, bootstrap, compare, holm, metrics_from_confusion, patient_confusions,
                              resample_indices, run_levels, two_sided_p)
from ramscore.metrics import compute_metrics

GRADES = [0, 1, 2, 3, 5]


def fake_run(rng, gt, keep):
    pred = np.where(rng.random(len(gt)) < keep, gt, rng.choice(GRADES, size=len(gt)))
    return pd.DataFrame({'patient': np.repeat([f'P{i:02d}' for i in range(30)], 20), 'gt': gt, 'pred': pred})


@pytest.fixture
def gt():
    return np.random.default_rng(0).choice(GRADES, size=600, p=[0.7, 0.15, 0.1, 0.03, 0.02])


def test_confusion_metrics_match_compute_metrics(gt):
    df = fake_run(np.random.default_rng(1), gt, keep=0.6)
    cm = patient_confusions(df, sorted(df['patient'].unique())).sum(0)
    fast, ref = metrics_from_confusion(cm), compute_metrics(df['gt'], df['pred'])
    for k, v in fast.items():
        assert float(v) == pytest.approx(ref[k]), k


def test_identical_runs_give_zero_difference(gt):
    rng = np.random.default_rng(2)
    runs = [fake_run(rng, gt, keep=0.6) for _ in range(3)]
    res = compare({'a': runs, 'b': [r.copy() for r in runs]}, 'a', n_boot=200)
    c = res['vs_baseline']['b-a']['bacc4']
    assert c['diff'] == 0 and c['ci_fixed'] == [0, 0]
    assert c['p_fixed'] == 1.0


def test_clearly_better_strategy_is_detected(gt):
    rng = np.random.default_rng(3)
    res = compare({'base': [fake_run(rng, gt, 0.3) for _ in range(3)],
                   'good': [fake_run(rng, gt, 0.9) for _ in range(3)]}, 'base', n_boot=500)
    c = res['vs_baseline']['good-base']
    assert c['bacc4']['ci_hier'][0] > 0 and c['bacc4']['p_holm'] < 0.01
    assert c['mae']['prob_better'] == 1.0 and c['bacc4']['seed_pairs_better'] == 3


def test_holm_and_p_value():
    assert holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert two_sided_p(np.ones(99)) == pytest.approx(0.02)


def test_run_levels_read_draw_and_seed_from_names():
    assert run_levels(['e2_a1_b0_f0.1_d0_s42', 'e2_a1_b0_f0.1_d1_s7']) == [(0, 42), (1, 7)]
    assert run_levels(['d2_s1337']) == [(2, 1337)]
    assert run_levels(['b0_resnet34_f1.0_s42']) == [(0, 42)]
    assert run_levels(['linear_f0.1_d0', 'linear_f0.1_d1']) == [(0, 0), (1, 0)]
    assert run_levels(['ab', 'cd']) == [(0, 0), (0, 1)]


def test_resample_indices_keep_seeds_inside_their_draw():
    levels = [(d, s) for d in range(3) for s in (42, 1337, 2026)]
    idx = resample_indices(levels, 2000, np.random.default_rng(0))
    assert idx.shape == (2000, 9)
    runs_of = np.array(sorted(range(9), key=lambda i: levels[i])).reshape(3, 3)
    draw_of = np.zeros(9, int)
    for d in range(3):
        draw_of[runs_of[d]] = d
    slots = draw_of[idx].reshape(2000, 3, 3)
    assert (slots == slots[:, :, :1]).all()                  # each slot of 3 seeds comes from one draw
    assert (slots[:, :, 0].min(1) == slots[:, :, 0].max(1)).mean() > 0.05   # a draw can be drawn repeatedly


def test_resample_indices_handle_flat_and_single_level_cells():
    rng = np.random.default_rng(1)
    assert set(np.unique(resample_indices([(0, 42), (0, 7), (0, 9)], 500, rng))) == {0, 1, 2}
    assert resample_indices([(0, 0), (1, 0), (2, 0)], 500, rng).shape == (500, 3)
    assert resample_indices([(0, 0, 1)], 5, rng).tolist() == [[0]] * 5


def test_resample_indices_reject_unbalanced_designs():
    with pytest.raises(ValueError):
        resample_indices([(0, 1), (0, 2), (1, 1)], 10, np.random.default_rng(0))


def test_run_picks_share_a_design_and_repeat_for_the_same_seed():
    levels = [(d, s) for d in range(3) for s in range(3)]
    a, b = RunPicks(100, 5), RunPicks(100, 5)
    assert a(levels) is a(levels)
    assert (a(levels) == b(levels)).all()


def test_identical_arms_stay_paired_under_run_resampling(gt):
    """The hierarchical interval is 0 for identical arms only if both arms redraw the same runs."""
    rng = np.random.default_rng(4)
    runs = [fake_run(rng, gt, keep=k) for k in (0.3, 0.6, 0.9) for _ in range(3)]
    levels = [(d, s) for d in range(3) for s in (42, 1337, 2026)]
    res = compare({'a': runs, 'b': [r.copy() for r in runs]}, 'a', n_boot=200, levels=levels)
    assert res['vs_baseline']['b-a']['bacc4']['ci_hier'] == [0, 0]


def test_between_draw_variance_is_part_of_the_hierarchical_interval(gt):
    """Draws that differ widen the interval of a mean; seeds nested in them do not hide that."""
    rng = np.random.default_rng(5)
    runs = [fake_run(rng, gt, keep=k) for k in (0.2, 0.5, 0.9) for _ in range(3)]
    other = [fake_run(rng, gt, keep=0.5) for _ in range(9)]
    levels = [(d, s) for d in range(3) for s in range(3)]
    nested = compare({'o': other, 'r': runs}, 'o', n_boot=1000, levels=levels)['ci_fixed']
    assert nested is not None
    point, _, hier, _, _ = bootstrap({'r': runs}, 1000, 0, levels)
    flat = bootstrap({'r': runs}, 1000, 0, None)[2]
    assert np.nanstd(hier['r']['bacc4']) > np.nanstd(flat['r']['bacc4'])
