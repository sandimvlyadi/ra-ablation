from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ramscore import e8
from ramscore.data import AUGMENTATIONS, get_transform, seed_augmentation_worker
from ramscore.train import get_args


class Noise(Dataset):
    """Same image at every index, so any difference between items comes from the augmentation."""

    def __init__(self, transform):
        self.transform = transform
        self.img = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)

    def __len__(self):
        return 48

    def __getitem__(self, idx):
        return torch.from_numpy(self.transform(image=self.img)['image'].copy())


def batches(aug='base', seed=7, workers=3):
    ds = Noise(get_transform('train', aug, seed))
    loader = DataLoader(ds, batch_size=4, num_workers=workers, worker_init_fn=partial(seed_augmentation_worker, seed))
    return [b.numpy() for b in loader]


def names(transform):
    return [type(t).__name__ for t in transform.transforms]


def test_arms_hold_the_right_steps():
    assert names(get_transform('train')) == ['ShiftScaleRotate', 'RandomBrightnessContrast']
    assert names(get_transform('train', 'base')) == ['ShiftScaleRotate', 'RandomBrightnessContrast']
    assert names(get_transform('train', 'geom')) == ['ShiftScaleRotate']
    assert names(get_transform('train', 'intensity')) == ['RandomBrightnessContrast']
    assert get_transform('train', 'none') is None


def test_only_training_is_augmented():
    for aug in AUGMENTATIONS:
        assert get_transform('val', aug) is None
        assert get_transform('test', aug) is None


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError):
        get_transform('train', 'mixup')


def test_seeded_loader_repeats_exactly():
    for one, two in zip(batches(), batches()):
        assert np.array_equal(one, two)


def test_other_seed_gives_other_augmentation():
    assert not all(np.array_equal(a, b) for a, b in zip(batches(seed=7), batches(seed=8)))


def test_workers_do_not_share_a_stream():
    per_worker = batches(workers=3)[:3]     # the first batch of each of the 3 workers
    assert not np.array_equal(per_worker[0], per_worker[1])
    assert not np.array_equal(per_worker[1], per_worker[2])


def test_defaults_keep_the_old_recipe():
    args = get_args(['--name', 'x'])
    assert args.aug == 'base' and args.seed_aug is False


def write_run(root, exp, name, shift, n=120):
    d = root / exp / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    gt = np.tile([0, 0, 0, 1, 1, 2, 3, 5], n // 8)
    pred = np.where(rng.random(len(gt)) < 0.5 + shift, gt, 0)
    joints = ['MCP-T', 'MCP-I', 'MCP-M', 'MCP-R']
    pd.DataFrame({'case': [f'c{i % 12}' for i in range(len(gt))], 'patient': [f'P{i % 12}' for i in range(len(gt))],
                  'joint': [joints[i % 4] for i in range(len(gt))], 'gt': gt, 'pred': pred}
                 ).to_csv(d / 'test_predictions.csv', index=False)


@pytest.fixture
def e8_runs(tmp_path):
    designs = {'1.0': [(0, 42), (0, 1337), (0, 2026)], '0.1': [(0, 42), (1, 42), (2, 42)]}
    for frac, tags in designs.items():
        for d, s in tags:
            for arm, shift in (('base', 0.1), ('none', -0.1), ('geom', 0.1), ('intensity', 0.1)):
                write_run(tmp_path, 'e8', f'e8_{arm}_a1_resnet34_b2_f{frac}_d{d}_s{s}', shift)
            write_run(tmp_path, 'e3', f'e3_a1_resnet34_b2_f{frac}_d{d}_s{s}', 0.1)
    return tmp_path


def test_primary_family_has_six_contrasts_with_holm(e8_runs):
    res = e8.analyse(e8_runs, n_boot=100)
    assert {(r['frac'], r['arm']) for r in res['primary']} == {(f, a) for f in ('1.0', '0.1') for a in e8.ARMS}
    assert all('p_holm' in r['bacc4'] and r['runs'] == 3 for r in res['primary'])
    assert len(res['supporting']) == 2 and all('p_holm' not in r['bacc4'] for r in res['supporting'])
    assert len(res['predictions']) == 8


def test_analysis_waits_for_the_runs(tmp_path):
    res = e8.analyse(tmp_path, n_boot=50)
    assert res == {'primary': [], 'supporting': [], 'predictions': []}


@pytest.mark.parametrize('diff, p, expected', [(-0.06, 0.01, 'meaningful'), (0.05, 0.04, 'meaningful'),
                                               (-0.03, 0.01, 'small'), (-0.09, 0.05, 'not shown')])
def test_verdict_applies_the_locked_rules(diff, p, expected):
    assert e8.verdict({'diff': diff, 'p_holm': p}) == expected


def test_gap_summary_and_report(e8_runs):
    rows = [{'run': f'r{i}', 'arm': 'none', 'frac': '1.0', 'best_epoch': 10 + i, 'train_bacc4': 0.9, 'val_bacc4': 0.5,
             'train_qwk': 0.8, 'val_qwk': 0.4, 'loss_last': 0.1} for i in range(3)]
    gaps = e8.summarise_gap(rows)
    assert gaps[0]['runs'] == 3 and gaps[0]['gap_bacc4'] == pytest.approx(0.4)
    text = e8.report(e8.analyse(e8_runs, n_boot=100), gaps)
    assert 'Post hoc' in text and 'Train-validation gap' in text and '`none`' in text
