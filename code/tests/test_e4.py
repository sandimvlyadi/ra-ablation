import numpy as np
import pandas as pd

from ramscore.e4 import LEVELS, analyse, fewshot_analysis, fewshot_patients, leave_one_out, to_levels
from ramscore.rheumavit import crop_square, hand_image, reference_scores


def test_hand_image_puts_both_thumbs_left():
    img = np.arange(12).reshape(2, 6)
    assert hand_image(img, 'left').tolist() == [[3, 4, 5], [9, 10, 11]]
    assert hand_image(img, 'right').tolist() == [[2, 1, 0], [8, 7, 6]]


def test_crop_square_pads_outside_with_black():
    img = np.full((10, 10), 7, np.uint8)
    c = crop_square(img, 1, 1, 4)
    assert c.shape == (4, 4) and c[0, 0] == 0 and c[3, 3] == 7


def test_reference_is_lower_median_of_available_readers():
    rows = [(1, 'left', 'MCP', 2.0, e, s) for e, s in ((0, 1), (1, 3), (2, 2))]
    rows += [(1, 'left', 'PIP', 3.0, e, s) for e, s in ((1, 0), (2, 2))]          # two readers disagree
    rows += [(1, 'left', 'DIP', 2.0, 1, 5)]                                         # not a shared joint
    scores = pd.DataFrame(rows, columns=['patient_id', 'hand', 'joint', 'finger', 'expert', 'score'])
    scores['disease'] = 'erosion'
    ref = reference_scores(scores).set_index('stem')
    assert ref.loc['MCP-I', 'score'] == 2 and ref.loc['MCP-I', 'n_readers'] == 3
    assert ref.loc['PIP-M', 'score'] == 0 and ref.loc['PIP-M', 'n_readers'] == 2
    assert 'DIP' not in ref.index and len(ref) == 2


def test_to_levels_keeps_shared_joints_and_merges_high_grades():
    df = pd.DataFrame({'patient': ['a'] * 3, 'joint': ['MCP-I', 'L', 'R'], 'gt': [4, 5, 1], 'pred': [5, 0, 3]})
    out = to_levels(df)
    assert out['gt'].tolist() == [3, 1] and out['pred'].tolist() == [3, 3]


def fake_runs(rng, keep_ext, n_runs=3):
    def df(keep, n_pat):
        gt = rng.choice(LEVELS, size=n_pat * 20, p=[0.6, 0.2, 0.12, 0.08])
        pred = np.where(rng.random(len(gt)) < keep, gt, rng.choice(LEVELS, size=len(gt)))
        return pd.DataFrame({'patient': np.repeat([f'P{i}' for i in range(n_pat)], 20), 'gt': gt, 'pred': pred})
    return [(df(0.9, 30), df(keep_ext, 25)) for _ in range(n_runs)]


def test_analyse_finds_the_smaller_drop():
    rng = np.random.default_rng(0)
    runs = {'K1': fake_runs(rng, 0.5), 'K2': fake_runs(rng, 0.2), 'K3': fake_runs(rng, 0.85), 'K4': fake_runs(rng, 0.85)}
    point, ci, per_run, contrasts = analyse(runs, n_boot=500)
    assert point['K3']['drop']['bacc4'] < point['K2']['drop']['bacc4']
    a0 = contrasts[0]
    assert a0['contrast'] == 'drop(K2) - drop(K3)' and a0['bacc4_diff'] > 0 and a0['bacc4_ci_hier'][0] > 0
    assert contrasts[1]['bacc4_p_holm'] > 0.05


def test_fewshot_patients_are_nested_and_cover_every_worst_grade(tmp_path):
    worst = np.repeat([0, 1, 2, 3], 25)
    m = pd.DataFrame({'patient': [f'P{i}' for i in range(100)], 'score': worst, 'e4_split': 'pool'})
    m = pd.concat([m, pd.DataFrame({'patient': ['T0'], 'score': [5], 'e4_split': ['test']})])
    m.to_csv(tmp_path / 'manifest.csv', index=False)
    for draw in (0, 1):
        p = fewshot_patients(tmp_path / 'manifest.csv', draw)
        assert len(set(p[20])) == 20 and set(p[5]) <= set(p[20]) and 'T0' not in p[20]
        grades = pd.Series(worst, index=m['patient'][:100])
        assert grades[p[20]].value_counts().to_dict() == {0: 5, 1: 5, 2: 5, 3: 5}
        assert set(grades[p[5]]) >= {0, 3}
    assert fewshot_patients(tmp_path / 'manifest.csv', 0) != fewshot_patients(tmp_path / 'manifest.csv', 1)


def test_fewshot_analysis_contrasts_per_size():
    rng = np.random.default_rng(0)
    keep = {'K1': 0.5, 'K2': 0.2, 'K3': 0.8, 'K4': 0.8}
    cells = {(c, k): [r[1] for r in fake_runs(rng, keep[c] + 0.05 * (k > 0), n_runs=3 if k == 0 else 9)]
             for c in keep for k in (0, 5, 20)}
    point, sd, ci, contrasts = fewshot_analysis(cells, n_boot=300)
    assert len(contrasts) == 6 and point[('K3', 5)]['bacc4'] > point[('K2', 5)]['bacc4']
    k5 = [r for r in contrasts if r['k'] == 5]
    assert k5[0]['contrast'] == 'K3 - K2' and k5[0]['bacc4_ci_hier'][0] > 0


def test_leave_one_out_reference_uses_the_other_readers():
    m = pd.DataFrame({'case': ['a', 'b', 'c', 'd'], 'patient': 'P', 'joint': 'MCP-I',
                      'r0': [1, np.nan, 3, 2], 'r1': [0, 2, np.nan, np.nan], 'r2': [2, 3, 1, np.nan]})
    loo = leave_one_out(m, 0).set_index('case')
    assert loo['ref'].to_dict() == {'a': 0, 'c': 1}          # 'b' unscored by reader 0, 'd' has no other reader
    loo = leave_one_out(m, 2).set_index('case')
    assert loo['ref'].to_dict() == {'a': 0, 'b': 2, 'c': 3}


def test_fewshot_levels_nest_source_runs_in_few_shot_draws():
    from ramscore.e4 import DRAWS, fewshot_levels
    levels = fewshot_levels(3 * len(DRAWS))
    assert len(levels) == 3 * len(DRAWS) and len(set(levels)) == len(levels)
    assert [d for d, _ in levels[:3]] == [DRAWS[0]] * 3
