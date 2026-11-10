import numpy as np
import pandas as pd

from ramscore.icc import bootstrap_cells, contrasts, hand_totals, icc_a1, icc_c1


def icc_reference(x1, x2):
    """ICC(A,1) from the two-way ANOVA mean squares (McGraw & Wong 1996)."""
    x = np.column_stack([x1, x2]).astype(float)
    n, k = x.shape
    m = x.mean()
    msr = k * ((x.mean(1) - m) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - m) ** 2).sum() / (k - 1)
    mse = ((x - x.mean(1, keepdims=True) - x.mean(0) + m) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def test_icc_matches_anova_and_frequency_weights_match_repeated_rows():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 40, 50)
    b = a + rng.integers(-5, 8, 50)
    assert np.isclose(icc_a1(a, b)[0], icc_reference(a, b))
    w = rng.integers(0, 3, 50)
    assert np.isclose(icc_a1(a, b, w)[0], icc_reference(np.repeat(a, w), np.repeat(b, w)))
    assert np.isclose(icc_a1(a, a)[0], 1.0)


def test_hand_totals_sum_native_grades_per_case():
    df = pd.DataFrame({'case': ['h1', 'h1', 'h2'], 'patient': ['p', 'p', 'p'], 'joint': ['IP', 'L', 'IP'],
                       'gt': [5, 3, 1], 'pred': [3, 0, 2]})
    t = hand_totals(df)
    assert t.loc['h1', 'gt'] == 8 and t.loc['h1', 'pred'] == 3
    assert hand_totals(df, joints=('IP',)).loc['h1', 'gt'] == 5


def test_bootstrap_contrast_favours_the_closer_model():
    rng = np.random.default_rng(1)
    gt = rng.integers(0, 30, 120)
    base = pd.DataFrame({'patient': np.repeat(np.arange(60), 2), 'gt': gt}, index=[f'c{i:03d}' for i in range(120)])
    good = [base.assign(pred=gt + rng.integers(-2, 3, 120)) for _ in range(3)]
    bad = [base.assign(pred=rng.integers(0, 30, 120)) for _ in range(3)]
    point, draws, _ = bootstrap_cells({'good': good, 'bad': bad}, n_boot=500)
    assert point['good'] > 0.9 > 0.3 > point['bad']
    row = contrasts('t', point, draws, [('good - bad', {'good': 1, 'bad': -1})])[0]
    assert row['ci'][0] > 0 and row['p_holm'] < 0.01


def test_icc_c1_ignores_a_constant_offset_but_icc_a1_does_not():
    a = np.arange(30) % 7 * 3.0
    assert np.isclose(icc_c1(a, a + 10), 1.0) and icc_a1(a, a + 10)[0] < 0.5


def _totals(rng, noise, n=40):
    """Same hands and reference in every run; only the prediction noise differs."""
    gt = np.random.default_rng(99).integers(0, 40, n).astype(float)
    return pd.DataFrame({'patient': [f'P{i // 2}' for i in range(n)], 'gt': gt, 'pred': gt + rng.normal(0, noise, n)})


def test_identical_cells_stay_paired_under_run_resampling():
    rng = np.random.default_rng(0)
    runs = [_totals(rng, noise) for noise in (2, 8, 20) for _ in range(3)]
    levels = [(d, s) for d in range(3) for s in range(3)]
    _, draws, _ = bootstrap_cells({'a': runs, 'b': [r.copy() for r in runs]}, n_boot=200, levels={'a': levels, 'b': levels})
    assert (draws['a'] == draws['b']).all()


def test_replicate_is_the_mean_over_runs_not_a_single_run():
    rng = np.random.default_rng(1)
    runs = [_totals(rng, noise) for noise in (2, 30)]
    _, draws, _ = bootstrap_cells({'a': runs}, n_boot=2000)
    low, high = sorted(icc_a1(r['gt'], r['pred'])[0] for r in runs)
    # Redrawing 2 runs gives (low + high) / 2 in half of the replicates; picking one run never lands near that
    middle = np.abs(draws['a'] - (low + high) / 2) < (high - low) / 4
    assert middle.mean() > 0.2


def test_runs_redraw_the_same_way_across_calls_with_one_pick_seed():
    rng = np.random.default_rng(2)
    runs = [_totals(rng, n) for n in (2, 10, 25)]
    a = bootstrap_cells({'x': runs}, n_boot=500, seed=0)[1]['x']
    b = bootstrap_cells({'x': runs}, n_boot=500, seed=0)[1]['x']
    assert (a == b).all()
