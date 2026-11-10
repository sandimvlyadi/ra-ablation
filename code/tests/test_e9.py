import json

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from ramscore import e9
from ramscore.data import VisitPairDataset, get_transform, visit_pairs
from ramscore.models import Criterion, expected_grade, monotonic_penalty
from ramscore.train import check_pair_args, get_args, paired_loss


def case(date, side='L', patient='P0001', centre='SARC'):
    return f'JP_{centre}_{patient}_{date}_1234_{side}'


def record(date, joint='R', side='L', patient='P0001', centre='SARC', **extra):
    return {'case': case(date, side, patient, centre), 'patient': f'{centre}_{patient}', 'joint': joint, **extra}


# ---------- visit pairs ----------

def test_pairs_are_consecutive_and_in_date_order():
    records = [record('20120101'), record('20100101'), record('20110101')]
    assert visit_pairs(records) == [(1, 2), (2, 0)]


def test_pairs_stay_within_joint_side_and_patient():
    records = [record('20100101'), record('20110101', joint='U'), record('20110101', side='R'),
               record('20110101', patient='P0002'), record('20110101', centre='HMCRD')]
    assert visit_pairs(records) == []


def test_same_date_images_are_not_ordered_but_both_reach_the_next_visit():
    records = [record('20100101'), record('20100101'), record('20110101')]
    assert visit_pairs(records) == [(0, 2), (1, 2)]


def test_pairs_read_no_label():
    assert all('score' not in r for r in [record('20100101'), record('20110101')])
    assert visit_pairs([record('20100101'), record('20110101')]) == [(0, 1)]


@pytest.fixture
def roi_records(tmp_path):
    out = []
    for value, date in ((10, '20100101'), (200, '20110101'), (90, '20100101')):
        path = tmp_path / f'{value}.png'
        cv2.imwrite(str(path), np.full((32, 32), value, dtype=np.uint8))
        joint = 'U' if value == 90 else 'R'
        out.append(record(date, joint=joint, img_path=str(path)))
    return out


def test_pair_dataset_returns_two_images_in_order_and_no_label(roi_records):
    ds = VisitPairDataset(roi_records, image_size=32)
    assert len(ds) == 1
    item = ds[0]
    assert set(item) == {'early', 'late'}
    assert item['early'].shape == (1, 32, 32)
    assert item['early'].mean().item() == pytest.approx(10 / 255)
    assert item['late'].mean().item() == pytest.approx(200 / 255)


def test_pair_dataset_caches_only_joints_in_a_pair(roi_records):
    assert len(VisitPairDataset(roi_records, image_size=32).cache) == 2


def test_pair_dataset_refuses_a_pool_without_pairs(roi_records):
    with pytest.raises(ValueError):
        VisitPairDataset(roi_records[:1], image_size=32)


def test_pair_dataset_applies_the_transform_separately_to_each_image(roi_records):
    ds = VisitPairDataset(roi_records, image_size=32, transform=get_transform('train', 'base', 7))
    assert ds.transform is not None
    assert ds[0]['early'].shape == (1, 32, 32)


# ---------- penalty ----------

def logits_for(probs):
    return torch.log(torch.tensor(probs, dtype=torch.float32))


def test_expected_grade_uses_class_positions():
    logits = logits_for([[.5, .5, 1e-9], [1e-9, 1e-9, 1.]])
    assert expected_grade(logits).tolist() == pytest.approx([.5, 2.], abs=1e-4)


def test_penalty_is_zero_when_the_later_visit_is_not_lower():
    early, late = logits_for([[1., 1e-9, 1e-9]]), logits_for([[1e-9, 1., 1e-9]])
    assert monotonic_penalty(early, late, 0.25).item() == pytest.approx(0, abs=1e-6)


def test_penalty_fires_only_beyond_the_margin():
    early, late = logits_for([[1e-9, 1e-9, 1.]]), logits_for([[1., 1e-9, 1e-9]])
    assert monotonic_penalty(early, late, 0.25).item() == pytest.approx(1.75, abs=1e-3)
    small_drop_early, small_drop_late = logits_for([[.9, .1, 1e-9]]), logits_for([[1., 1e-9, 1e-9]])
    assert monotonic_penalty(small_drop_early, small_drop_late, 0.25).item() == pytest.approx(0, abs=1e-5)


def test_penalty_sends_gradient_to_both_visits():
    early = logits_for([[1e-3, 1e-3, 1.]]).requires_grad_()
    late = logits_for([[1., 1e-3, 1e-3]]).requires_grad_()
    monotonic_penalty(early, late, 0.).backward()
    assert early.grad.abs().sum() > 0 and late.grad.abs().sum() > 0


# ---------- training step ----------

class ByValue(nn.Module):
    """Output depends only on the input value, so a wrong split of the batch changes the numbers."""

    def forward(self, x):
        v = x.flatten(1).mean(1)
        return torch.stack([v, 2 * v, 3 * v], 1)


def test_paired_loss_splits_the_batch_correctly():
    img = torch.full((2, 1, 2, 2), .1)
    target = torch.tensor([0, 1])
    pair = {'early': torch.full((3, 1, 2, 2), .9), 'late': torch.full((3, 1, 2, 2), .2)}
    net, crit = ByValue(), Criterion('softmax', np.array([0, 1, 2]), 3)
    sup, penalty, violated = paired_loss(net, img, target, pair, crit, 0.)
    assert sup.item() == pytest.approx(crit(net(img), target).item())
    assert penalty.item() == pytest.approx(monotonic_penalty(net(pair['early']), net(pair['late']), 0.).item())
    assert violated == pytest.approx(1.0)


def test_defaults_keep_the_old_recipe():
    args = get_args(['--name', 'x'])
    assert args.pair_weight == 0 and args.pair_margin == 0.25 and args.pair_batch == 16
    check_pair_args(args)


@pytest.mark.parametrize('extra', [
    ['--head', 'ordinal'], ['--head', 'corn'], ['--levels', '3'], ['--mask-root', 'x'], ['--undersample-zero', 'rotate'],
    ['--init-from', 'x', '--freeze-backbone'],
])
def test_pair_penalty_refuses_untested_combinations(extra):
    base = ['--name', 'x', '--pair-weight', '.3', '--head', 'softmax']
    with pytest.raises(SystemExit):
        check_pair_args(get_args(base + extra))


def test_pair_penalty_accepts_the_e9_recipe():
    check_pair_args(get_args(['--name', 'x', '--pair-weight', '.3', '--head', 'softmax', '--logit-adjust', '1']))


# ---------- weight selection ----------

def write_run(path, bacc4, qwk, test_bacc4=.9):
    path.mkdir(parents=True)
    (path / 'metrics.json').write_text(json.dumps({'val': {'bacc4': bacc4, 'qwk': qwk}, 'test': {'bacc4': test_bacc4}}))


def tune_tree(tmp_path, cells, base=(.40, .30)):
    write_run(tmp_path / 'e8' / 'e8_base_a1_resnet34_b2_f0.1_d0_s42', *base)
    for weight, (b, q) in cells.items():
        write_run(tmp_path / 'e9' / f'e9_l{weight}_a1_resnet34_b2_f0.1_d0_s42', b, q)
    return tmp_path


def test_selection_takes_highest_val_bacc4_above_the_qwk_floor(tmp_path):
    root = tune_tree(tmp_path, {'0.1': (.45, .30), '0.3': (.50, .29), '1.0': (.60, .20)})
    assert e9.select_weight(root, '0.1')['weight'] == '0.3'


def test_selection_breaks_ties_on_val_qwk(tmp_path):
    root = tune_tree(tmp_path, {'0.1': (.45, .30), '0.3': (.45, .35), '1.0': (.40, .30)})
    assert e9.select_weight(root, '0.1')['weight'] == '0.3'


def test_selection_falls_back_to_highest_qwk_and_says_so(tmp_path):
    root = tune_tree(tmp_path, {'0.1': (.45, .10), '0.3': (.50, .20), '1.0': (.60, .05)})
    choice = e9.select_weight(root, '0.1')
    assert choice['weight'] == '0.3' and choice['floor_met'] is False


def test_selection_ignores_the_test_set(tmp_path):
    root = tune_tree(tmp_path, {'0.1': (.45, .30), '0.3': (.45, .30), '1.0': (.45, .30)})
    (root / 'e9' / 'e9_l1.0_a1_resnet34_b2_f0.1_d0_s42' / 'metrics.json').write_text(
        json.dumps({'val': {'bacc4': .45, 'qwk': .30}, 'test': {'bacc4': .99}}))
    assert e9.select_weight(root, '0.1')['weight'] == '0.1'


def test_selection_waits_for_all_three_weights(tmp_path):
    root = tune_tree(tmp_path, {'0.1': (.45, .30)})
    with pytest.raises(FileNotFoundError):
        e9.select_weight(root, '0.1')


# ---------- verdict ----------

def stats(diff, p_holm=1.0):
    return {'diff': diff, 'p_holm': p_holm}


@pytest.mark.parametrize('bacc, qwk, auc, decline, expected', [
    (stats(.06, .01), stats(.0), 0., -.01, 'pass'),
    (stats(.0), stats(.06, .01), 0., -.01, 'pass'),
    (stats(.06, .2), stats(.0), 0., -.01, 'promising'),
    (stats(.04, .01), stats(.0), 0., -.01, 'no gain'),
    (stats(.06, .01), stats(-.03), 0., -.01, 'no gain'),
    (stats(.06, .01), stats(.0), -.03, -.01, 'no gain'),
    (stats(.06, .01), stats(.0), 0., .01, 'no gain'),
    (stats(-.06, .01), stats(.06, .01), 0., -.01, 'harm'),
    (stats(.0), stats(-.05, .04), 0., -.01, 'harm'),
    (stats(-.06, .2), stats(.0), 0., -.01, 'no gain'),
])
def test_verdict_applies_the_locked_rules(bacc, qwk, auc, decline, expected):
    assert e9.verdict(bacc, qwk, auc, decline) == expected


# ---------- prediction summaries ----------

def frame(rows):
    return pd.DataFrame(rows, columns=['case', 'patient', 'joint', 'gt', 'pred'])


def test_decline_share_counts_falls_between_consecutive_visits():
    df = frame([(case('20100101'), 'SARC_P0001', 'R', 0, 2), (case('20110101'), 'SARC_P0001', 'R', 0, 1),
                (case('20120101'), 'SARC_P0001', 'R', 0, 3), (case('20100101'), 'SARC_P0001', 'U', 0, 0)])
    assert e9.decline_share(df, 'pred') == pytest.approx(.5)
    assert e9.decline_share(df, 'gt') == 0


def test_monotone_subset_drops_series_whose_reference_falls():
    df = frame([(case('20100101'), 'SARC_P0001', 'R', 2, 2), (case('20110101'), 'SARC_P0001', 'R', 1, 1),
                (case('20100101'), 'SARC_P0001', 'U', 0, 0), (case('20110101'), 'SARC_P0001', 'U', 1, 1),
                (case('20100101'), 'SARC_P0001', 'L', 1, 1)])
    kept = e9.monotone_subset(df)
    assert sorted(kept['joint'].unique()) == ['U']


def test_within_joint_auroc_ignores_differences_between_joints():
    df = frame([('a', 'p', 'R', 0, 3), ('b', 'p', 'R', 1, 5), ('c', 'p', 'U', 0, 0), ('d', 'p', 'U', 1, 1)])
    assert e9.within_joint_auroc(df) == pytest.approx(1.)


def test_no_image_baseline_uses_joint_type_only():
    high = [2] * 10 + [3] * 10
    train = pd.DataFrame({'joint': ['R'] * 80 + ['U'] * 80,
                          'score': [0] * 30 + [1] * 30 + high + [0] * 57 + [1] * 3 + high})
    target = pd.DataFrame({'joint': ['R', 'U'], 'score': [1, 0]})
    pred = e9.no_image_predictions(train, target)
    assert pred.tolist() == [1, 0]


# ---------- analysis ----------

def write_predictions(path, rng, shift=0):
    path.mkdir(parents=True)
    rows = []
    grades = [0, 0, 1, 2, 3, 5]
    for p in range(30):
        for visit, date in enumerate(('20100101', '20110101')):
            for j, joint in enumerate(('R', 'U')):
                gt = grades[(p + visit + j) % len(grades)]      # one reference for every run, as in real test splits
                pred = gt if rng.random() < .5 + shift else int(rng.choice([0, 1, 2, 3, 5]))
                rows.append((case(date, patient=f'P{p:04d}'), f'SARC_P{p:04d}', joint, gt, pred))
    frame(rows).to_csv(path / 'test_predictions.csv', index=False)


@pytest.fixture
def e9_runs(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / 'e9').mkdir()
    (tmp_path / 'e9' / 'choice.json').write_text(json.dumps({'0.1': {'weight': '0.3'}, '1.0': {'weight': '1.0'}}))
    tail = 'a1_resnet34_b2_f{f}_d{d}_s{s}'
    for s in e9.SEEDS:
        write_predictions(tmp_path / 'e8' / f"e8_base_{tail.format(f='1.0', d=0, s=s)}", rng)
        write_predictions(tmp_path / 'e9' / f"e9_l1.0_{tail.format(f='1.0', d=0, s=s)}", rng, .2)
        for d in e9.DRAWS:
            base = 'e8/e8_base' if s == 42 else 'e9/e9_base'
            write_predictions(tmp_path / f"{base}_{tail.format(f='0.1', d=d, s=s)}", rng)
            write_predictions(tmp_path / 'e9' / f"e9_l0.3_{tail.format(f='0.1', d=d, s=s)}", rng, .2)
    return tmp_path


def test_primary_family_has_four_contrasts_with_holm(e9_runs):
    results = e9.analyse(e9_runs, n_boot=20)
    assert [r['frac'] for r in results['primary']] == ['1.0', '0.1']
    assert [r['runs'] for r in results['primary']] == [3, 9]
    family = [m['p_holm'] for r in results['primary'] for m in (r['bacc4'], r['qwk'])]
    assert len(family) == 4 and all(0 <= p <= 1 for p in family)
    assert all(r['verdict'] in e9.VERDICTS for r in results['primary'])


def test_base_runs_come_from_both_sources(e9_runs):
    variant, base, _ = e9.paired_with_base(e9_runs, '0.1', '0.3')
    assert len(variant) == len(base) == 9


def test_analysis_waits_for_the_runs(tmp_path):
    (tmp_path / 'e9').mkdir()
    (tmp_path / 'e9' / 'choice.json').write_text(json.dumps({'0.1': {'weight': '0.3'}, '1.0': {'weight': '1.0'}}))
    assert e9.analyse(tmp_path, n_boot=5)['primary'] == []


def test_report_states_post_hoc_and_every_contrast(e9_runs):
    results = e9.analyse(e9_runs, n_boot=20)
    text = e9.report(results, {'0.1': {'bacc4': .3, 'qwk': 0.}, '1.0': {'bacc4': .32, 'qwk': 0.}})
    assert 'Post hoc' in text and '| 10% |' in text and '| 100% |' in text and 'No-image' in text
