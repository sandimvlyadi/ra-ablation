import numpy as np
import pandas as pd

from ramscore.e6 import analyse

GRADES = [0, 1, 2, 3, 5]


def write(root, prefix, tag, gt, keep, rng):
    d = root / f'{prefix}_b0_f0.1_{tag}'
    d.mkdir(parents=True)
    pred = np.where(rng.random(len(gt)) < keep, gt, rng.choice(GRADES, size=len(gt)))
    pd.DataFrame({'case': np.repeat([f'h{i:02d}' for i in range(40)], 10), 'patient': np.repeat(np.arange(20), 20),
                  'joint': np.tile([f'J{j}' for j in range(10)], 40), 'gt': gt, 'pred': pred}
                 ).to_csv(d / 'test_predictions.csv', index=False)


def test_e6_pairs_shared_tags_and_finds_the_better_recipe(tmp_path):
    rng = np.random.default_rng(0)
    gt = rng.choice(GRADES, size=400, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    for tag in ('d0_s42', 'd0_s1337', 'd1_s42'):
        write(tmp_path / 'e2', 'e2_a3_vits', tag, gt, 0.3, rng)
        write(tmp_path / 'e2', 'e2_a1_vits', tag, gt, 0.3, rng)
    for tag in ('d0_s42', 'd0_s1337'):                      # E6 still running: d1_s42 not there yet
        write(tmp_path / 'e6', 'e6_a3_vits', tag, gt, 0.9, rng)
    res = analyse(tmp_path, n_boot=300)
    cell = res[('0.1', 'b0')]
    assert cell['tags'] == ['d0_s1337', 'd0_s42']
    first = cell['contrasts'][0]
    assert first['contrast'] == 'mae_llrd - mae_e2' and first['bacc4']['diff'] > 0 and 'p_holm' in first['bacc4']
