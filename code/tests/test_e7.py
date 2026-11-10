import json
import math
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler

from ramscore.data import BE_JOINTS, RotatingZeroSampler, undersample_summary, undersample_zero, zero_quotas
from ramscore.e7 import analyse, parse_name, prediction_stats, report, verdict
from ramscore.train import get_args, main, make_loader

# Non-zero joints per joint type in the RAM-H1200 train split (793 of each type, 12,688 in total)
NONZERO = {'CMC-T': 115, 'IP': 141, 'L': 66, 'MCP-I': 154, 'MCP-M': 140, 'MCP-R': 120, 'MCP-S': 101, 'MCP-T': 133,
           'PIP-I': 284, 'PIP-M': 240, 'PIP-R': 303, 'PIP-S': 232, 'R': 90, 'S': 182, 'Tm': 189, 'U': 153}


@pytest.fixture(scope='module')
def train_records():
    records = []
    for joint, nz in NONZERO.items():
        for i in range(793):
            records.append({'joint': joint, 'score': 1 + i % 2 if i < nz else 0, 'patient': f'P{i % 154}',
                            'centre': 'A' if i % 2 else 'B', 'case': f'c{i}'})
    return records


def zero_ids(records):
    return {i for i, r in enumerate(records) if r['score'] == 0}


def test_real_counts_of_the_fixture(train_records):
    assert len(train_records) == 12688 and len(zero_ids(train_records)) == 10045


def test_quotas_are_one_to_one_with_one_retention_rate(train_records):
    quotas = zero_quotas(train_records)
    assert sum(quotas.values()) == 2643
    zeros = Counter(r['joint'] for r in train_records if r['score'] == 0)
    for joint, count in zeros.items():
        assert abs(quotas[joint] - count * 2643 / 10045) < 1


def test_quotas_do_not_trim_when_zeros_are_not_the_majority():
    records = [{'joint': 'R', 'score': 0}] * 3 + [{'joint': 'R', 'score': 2}] * 5 + [{'joint': 'U', 'score': 0}] * 1
    assert zero_quotas(records) == {'R': 3, 'U': 1}


def test_permanent_keeps_every_eroded_joint_and_the_quota_per_joint(train_records):
    kept = undersample_zero(train_records, seed=42)
    assert len(kept) == 5286
    assert sum(r['score'] > 0 for r in kept) == 2643
    assert Counter(r['joint'] for r in kept if r['score'] == 0) == Counter(zero_quotas(train_records))


def test_permanent_subset_follows_the_seed(train_records):
    picked = lambda seed: [id(r) for r in undersample_zero(train_records, seed)]
    assert picked(42) == picked(42)
    assert picked(42) != picked(1337)


def test_rotation_has_a_constant_composition_and_no_duplicates(train_records):
    sampler = RotatingZeroSampler(train_records, seed=42)
    quotas = zero_quotas(train_records)
    assert len(sampler) == 5286
    for _ in range(10):
        idx = list(sampler)
        assert len(idx) == len(set(idx)) == 5286
        assert sum(train_records[i]['score'] > 0 for i in idx) == 2643
        assert Counter(train_records[i]['joint'] for i in idx if train_records[i]['score'] == 0) == Counter(quotas)


def test_rotation_uses_every_zero_within_four_epochs_and_evens_out(train_records):
    sampler = RotatingZeroSampler(train_records, seed=42)
    zeros = zero_ids(train_records)
    seen = Counter()
    for epoch in range(50):
        seen.update(i for i in sampler if i in zeros)
        if epoch == 3:
            assert set(seen) == zeros
    for joint in NONZERO:
        counts = [seen[i] for i in zeros if train_records[i]['joint'] == joint]
        assert max(counts) - min(counts) <= 1


def test_rotation_changes_each_epoch_and_follows_the_seed(train_records):
    a, b = RotatingZeroSampler(train_records, seed=42), RotatingZeroSampler(train_records, seed=42)
    first = list(a)
    assert first == list(b)
    assert set(first) != set(a)
    assert first != list(RotatingZeroSampler(train_records, seed=1337))


class Indices(torch.utils.data.Dataset):
    num_classes = 5

    def __init__(self, records):
        self.records = records
        self.labels = np.array([min(r['score'], 4) for r in records])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return i


def loader_args(*extra):
    return get_args(['--name', 'x', '--workers', '0', *extra])


def test_rotating_loader_has_165_batches_and_min_steps_gives_120_epochs(train_records):
    args = loader_args('--undersample-zero', 'rotate')
    loader = make_loader(Indices(train_records), args, True, torch.Generator().manual_seed(0))
    assert len(loader) == 165
    assert max(50, math.ceil(19800 / len(loader))) == 120
    assert max(50, math.ceil(2000 / len(loader))) == 50


def test_rotation_advances_once_per_epoch_with_persistent_workers():
    records = [{'joint': 'R', 'score': 1 if i < 25 else 0} for i in range(100)]
    args = get_args(['--name', 'x', '--workers', '2', '--batch-size', '1', '--undersample-zero', 'rotate'])
    loader = make_loader(Indices(records), args, True, torch.Generator().manual_seed(0))
    seen, zeros = [], set(range(25, 100))
    for _ in range(3):
        epoch = [int(i) for batch in loader for i in batch]
        assert len(epoch) == len(set(epoch)) == 50
        seen.append(set(epoch) & zeros)
    assert not seen[0] & seen[1] and not seen[1] & seen[2] and seen[0] | seen[1] | seen[2] == zeros


@pytest.mark.parametrize('flags', [(), ('--undersample-zero', 'off')])
def test_off_leaves_the_baseline_loader_and_generator_untouched(train_records, flags):
    g = torch.Generator().manual_seed(7)
    before = g.get_state().clone()
    loader = make_loader(Indices(train_records), loader_args(*flags), True, g)
    assert isinstance(loader.sampler, RandomSampler) and len(loader) == 396
    assert torch.equal(g.get_state(), before)


def test_validation_loader_is_never_undersampled(train_records):
    loader = make_loader(Indices(train_records), loader_args('--undersample-zero', 'rotate'), False, torch.Generator())
    assert len(loader.dataset) == len(loader.sampler) == 12688


@pytest.mark.parametrize('mode', ['permanent', 'rotate'])
@pytest.mark.parametrize('extra', [('--class-weight',), ('--oversample', '1'), ('--logit-adjust', '1'),
                                   ('--init-from', 'x.pt', '--freeze-backbone')])
def test_inconsistent_combinations_are_refused(tmp_path, mode, extra):
    with pytest.raises(SystemExit):
        main(['--name', 'x', '--out', str(tmp_path), '--undersample-zero', mode, *extra])
    assert not (tmp_path / 'x').exists()


def test_summary_records_the_composition(train_records):
    s = undersample_summary(train_records, 'permanent', 42)
    assert s['mode'] == 'permanent' and s['seed'] == 42
    assert s['pool']['zero'] == 10045 and s['pool']['nonzero'] == 2643
    assert s['per_epoch'] == {'zero': 2643, 'nonzero': 2643, 'total': 5286}
    assert s['kept']['zero'] == 2643 and s['kept']['patients'] <= s['pool']['patients']
    assert s['zero_quota']['L'] == zero_quotas(train_records)['L']
    r = undersample_summary(train_records, 'rotate', 42)
    assert r['kept'] is None and r['per_epoch'] == s['per_epoch']


def make_dataset(root):
    """16 joints per hand on 4 hands per split; train has mostly grade 0, val and test hold all the grades."""
    stems = list(BE_JOINTS)
    for split, tag in (('train', 1), ('val', 2), ('test', 3)):
        (root / split).mkdir(parents=True)
        scores = {}
        for h in range(4):
            case = f'JP_SARC_P{tag}{h:03d}_20200101_{h:04d}_R'
            (root / split / case).mkdir()
            for k, stem in enumerate(stems):
                cv2.imwrite(str(root / split / case / f'{stem}.bmp'), np.full((24, 24), 10 + k, np.uint8))
            grades = [(0, 0, 0, 1)[k % 4] for k in range(16)] if split == 'train' else [(0, 1, 2, 3, 5)[k % 5] for k in range(16)]
            scores[f'{case}.bmp'] = {BE_JOINTS[s]: g for s, g in zip(stems, grades)}
        (root / split / '_annotation_be_scores.json').write_text(json.dumps(scores))


@pytest.mark.parametrize('mode', ['off', 'permanent', 'rotate'])
def test_training_end_to_end_keeps_validation_and_test_whole(tmp_path, mode):
    make_dataset(tmp_path / 'data')
    main(['--name', 'run', '--out', str(tmp_path / 'out'), '--data-root', str(tmp_path / 'data'), '--model', 'resnet18',
          '--image-size', '32', '--batch-size', '16', '--epochs', '1', '--workers', '0', '--n-boot', '0',
          '--undersample-zero', mode])
    run = tmp_path / 'out' / 'run'
    assert len(pd.read_csv(run / 'test_predictions.csv')) == 64
    assert (run / 'undersample.json').exists() == (mode != 'off')
    assert json.loads((run / 'args.json').read_text())['undersample_zero'] == mode
    assert len(pd.read_csv(run / 'history.csv')) == 1
    if mode != 'off':
        s = json.loads((run / 'undersample.json').read_text())
        assert s['pool']['zero'] == 48 and s['per_epoch']['total'] == 32


GRADES = [0, 1, 2, 3, 5]
SEEDS = (42, 1337, 2026)


def write_run(root, exp, name, gt, pred):
    d = root / exp / name
    d.mkdir(parents=True)
    pd.DataFrame({'case': np.repeat([f'h{i:02d}' for i in range(40)], 10), 'patient': np.repeat(np.arange(20), 20),
                  'joint': np.tile([f'J{j}' for j in range(10)], 40), 'gt': gt, 'pred': pred}
                 ).to_csv(d / 'test_predictions.csv', index=False)


@pytest.fixture
def runs(tmp_path):
    rng = np.random.default_rng(0)
    gt = rng.choice(GRADES, size=400, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    noisy = lambda keep: np.where(rng.random(400) < keep, gt, rng.choice(GRADES, size=400))
    for enc in ('a0_resnet34', 'a1_resnet34'):
        for seed in SEEDS:
            tag = f'{enc}_b0_f1.0_d0_s{seed}'
            write_run(tmp_path, 'e2', f'e2_{tag}', gt, noisy(0.2))
            write_run(tmp_path, 'e3', f'e3_{enc}_b2_f1.0_d0_s{seed}', gt, noisy(0.95))
            write_run(tmp_path, 'e7', f'e7_uperm_{tag}', gt, noisy(0.6))
            write_run(tmp_path, 'e7', f'e7_urot_{tag}', gt, noisy(0.6))
    for seed in SEEDS:
        for arm in ('upermlong', 'urotlong'):
            write_run(tmp_path, 'e7', f'e7_{arm}_a0_resnet34_b0_f1.0_d0_s{seed}', gt, noisy(0.6))
    return tmp_path


def test_parse_name():
    assert parse_name('e7_urotlong_a0_resnet34_b0_f1.0_d0_s1337') == {'arm': 'urotlong', 'enc': 'a0_resnet34', 'seed': 1337}
    assert parse_name('e6abl_mask_a0_resnet34_b2_f1.0_d0_s42') is None


def test_primary_family_has_eight_contrasts_with_holm(runs):
    res = analyse(runs, n_boot=200)
    primary = res['primary']
    assert len(primary) == 8
    assert {(r['arm'], r['enc'], r['base']) for r in primary} == {
        (a, e, b) for a in ('uperm', 'urot') for e in ('a0_resnet34', 'a1_resnet34') for b in ('b0', 'b2')}
    assert all('p_holm' in r['bacc4'] and r['runs'] == 3 and 'icc' in r for r in primary)
    by_base = {r['base']: r for r in primary}
    assert by_base['b0']['bacc4']['diff'] > 0 and by_base['b2']['bacc4']['diff'] < 0


def test_supporting_contrasts_are_not_in_the_primary_holm_family(runs):
    res = analyse(runs, n_boot=200)
    assert {(r['variant'], r['enc'], r['against']) for r in res['supporting']} == {
        *(('urot', e, 'uperm') for e in ('a0_resnet34', 'a1_resnet34')),
        ('upermlong', 'a0_resnet34', 'uperm'), ('urotlong', 'a0_resnet34', 'urot'),
        *((a, 'a0_resnet34', b) for a in ('upermlong', 'urotlong') for b in ('b0', 'b2'))}
    assert all('p_holm' not in r['bacc4'] for r in res['supporting'])


def test_analysis_waits_for_the_runs(tmp_path):
    res = analyse(tmp_path, n_boot=50)
    assert res['primary'] == [] and res['supporting'] == []


@pytest.mark.parametrize('base,diff,ci,p,expected', [
    ('b0', 0.05, (0.01, 0.09), 0.04, 'better'),
    ('b0', 0.049, (0.01, 0.09), 0.04, 'not shown'),        # below the minimum meaningful difference
    ('b0', 0.08, (0.01, 0.15), 0.06, 'not shown'),
    ('b2', 0.01, (0.001, 0.02), 0.04, 'better'),           # against B2 the CI rule applies, not the +5 points
    ('b2', 0.01, (-0.001, 0.02), 0.04, 'not shown'),
    ('b0', -0.06, (-0.1, -0.02), 0.01, 'worse'),
    ('b2', -0.06, (-0.1, -0.02), 0.01, 'worse'),
    ('b2', -0.06, (-0.1, -0.02), 0.2, 'not shown'),
])
def test_verdict_applies_the_locked_rules(base, diff, ci, p, expected):
    assert verdict(base, {'diff': diff, 'ci': list(ci), 'p_holm': p}) == expected


def test_prediction_stats_recall_and_grade_zero_share():
    df = pd.DataFrame({'gt': [0, 0, 1, 1, 2, 3, 5, 5], 'pred': [0, 1, 1, 0, 2, 0, 5, 3]})
    s = prediction_stats([df, df])
    assert s['pred_share_grade0'] == pytest.approx(3 / 8)
    assert s['true_share_grade0'] == pytest.approx(2 / 8)
    assert s['recall'] == pytest.approx({'0': 0.5, '1': 0.5, '2': 1.0, '3': 0.0, '5': 0.5})


def test_report_names_the_status_and_the_verdicts(runs):
    text = report(analyse(runs, n_boot=200))
    for needle in ('Post hoc', 'Verdict', 'Supporting', 'Grade 0 share'):
        assert needle in text
    assert any(f'| {v} |' in text for v in ('better', 'worse', 'not shown'))
