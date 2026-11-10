"""E9: penalty on a fall of the expected grade between visits, on ResNet34 + A1 ImageNet + B2, at 100% and 10% labels.

Consecutive visits of one joint come from every training patient, labelled or not, and inference takes a single image.
Post hoc: the test set was already used. Rules were fixed before the first run in papers/e9_prespecification_*.md.
Paired patient bootstrap of compare.py (hier: runs redrawn), Holm over BACC4 and QWK at both label fractions. The
comparator is the seeded `base` recipe of E8 plus e9_base for the seeds E8 did not run.

Usage: uv run --group analysis python -m ramscore.e9 choose     # pick the weights, validation only
       uv run --group analysis python -m ramscore.e9            # contrasts and report
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score

from .compare import holm, run_levels
from .data import LEVEL4, load_records, parse_case, subsample_patients
from .e6abl import contrast, load
from .e7 import prediction_stats
from .metrics import compute_metrics

WEIGHTS = ('0.1', '0.3', '1.0')
FRACTIONS = ('1.0', '0.1')
SEEDS = (42, 1337, 2026)
DRAWS = (0, 1, 2)
TAIL = 'a1_resnet34_b2_f{frac}_d{draw}_s{seed}'
QWK_FLOOR = 0.02
MIN_DIFF = 0.05
GUARD = 0.02
ALPHA = 0.05
VERDICTS = ('pass', 'promising', 'no gain', 'harm')
DATA_ROOT = '/data/Smart-X/tesis-fix/dataset/RAM-H1200-v1/SvdH_Scoring/SvdH_BE_Scoring'


def select_weight(root, frac, base_root=None):
    """Locked rule, on the design d0 s42 and on validation only: highest BACC4 among weights whose QWK stays within
    QWK_FLOOR of `base`, ties on QWK, then the smaller weight. If none does, the highest QWK, flagged."""
    root, tail = Path(root), TAIL.format(frac=frac, draw=0, seed=42)
    base_path = Path(base_root or root) / 'e8' / f'e8_base_{tail}' / 'metrics.json'
    base = json.loads(base_path.read_text())['val']
    cells = []
    for weight in WEIGHTS:
        val = json.loads((root / 'e9' / f'e9_l{weight}_{tail}' / 'metrics.json').read_text())['val']
        cells.append({'weight': weight, 'bacc4': val['bacc4'], 'qwk': val['qwk']})
    floor = base['qwk'] - QWK_FLOOR
    eligible = [c for c in cells if c['qwk'] >= floor]
    pick = max(eligible, key=lambda c: (c['bacc4'], c['qwk'])) if eligible else max(cells, key=lambda c: c['qwk'])
    return {'weight': pick['weight'], 'floor_met': bool(eligible), 'qwk_floor': floor,
            'base_val': {'bacc4': base['bacc4'], 'qwk': base['qwk']}, 'cells': cells}


def verdict(bacc4, qwk, auroc_diff, decline_diff):
    """Locked rules. bacc4 and qwk: dicts with diff and p_holm. auroc_diff, decline_diff: arm minus base, points."""
    metrics = (bacc4, qwk)
    if any(m['p_holm'] < ALPHA and m['diff'] <= -MIN_DIFF for m in metrics):
        return 'harm'
    guards = min(m['diff'] for m in metrics) >= -GUARD and auroc_diff >= -GUARD and decline_diff < 0
    if not guards:
        return 'no gain'
    if any(m['diff'] >= MIN_DIFF and m['p_holm'] < ALPHA for m in metrics):
        return 'pass'
    return 'promising' if any(m['diff'] >= MIN_DIFF for m in metrics) else 'no gain'


def _with_visits(df):
    cases = df['case'].map(lambda c: parse_case(c))
    return df.assign(side=[c.side for c in cases], date=[c.date for c in cases]).sort_values('date')


def decline_share(df, column='pred'):
    """Share of consecutive-visit joint pairs whose grade falls. Series keys hold patient, side and joint."""
    d = _with_visits(df)
    previous = d.groupby(['patient', 'side', 'joint'])[column].shift(1)
    known = previous.notna()
    return float((d.loc[known, column] < previous[known]).mean()) if known.any() else float('nan')


def monotone_subset(df):
    """Joints in a series of at least two visits whose reference never falls. Selected on the reference, so descriptive."""
    d = _with_visits(df)
    keys = ['patient', 'side', 'joint']
    previous = d.groupby(keys)['gt'].shift(1)
    d = d.assign(fall=d['gt'] < previous)
    grouped = d.groupby(keys)
    keep = (grouped['fall'].transform('max') == 0) & (grouped['gt'].transform('size') >= 2)
    return d.loc[keep, df.columns]


def within_joint_auroc(df):
    """Mean over joint types of the AUROC for any erosion, from the predicted grade. No credit for telling joints apart."""
    scores = [roc_auc_score(x['gt'] > 0, x['pred']) for _, x in df.groupby('joint') if 0 < (x['gt'] > 0).sum() < len(x)]
    return float(np.mean(scores)) if scores else float('nan')


def no_image_predictions(train, target):
    """Level per joint from the joint type alone: the training level prior of that joint over the overall prior."""
    levels = train['score'].map(LEVEL4)
    prior = pd.crosstab(train['joint'], levels).reindex(columns=range(4), fill_value=0) + 1
    prior = prior.div(prior.sum(axis=1), axis=0)
    overall = (levels.value_counts().reindex(range(4), fill_value=0) + 1) / len(train)
    return (prior / overall).reindex(target['joint']).to_numpy().argmax(axis=1)


def no_image_baseline(data_root=DATA_ROOT):
    """BACC4 and QWK on test of the no-image prediction, per label fraction (10%: mean over the patient draws)."""
    records = load_records(data_root, 'train')
    test = pd.DataFrame(load_records(data_root, 'test'))
    truth = test['score'].map(LEVEL4)
    out = {}
    for frac, draws in (('1.0', [0]), ('0.1', DRAWS)):
        bacc, qwk = [], []
        for draw in draws:
            train = pd.DataFrame(subsample_patients(records, float(frac), draw))
            pred = no_image_predictions(train, test)
            bacc.append(balanced_accuracy_score(truth, pred))
            qwk.append(cohen_kappa_score(truth, pred, weights='quadratic'))
        out[frac] = {'bacc4': float(np.mean(bacc)), 'qwk': float(np.mean(qwk))}
    return out


def base_sources(root, frac):
    """Seeded `base` runs by (draw, seed) tag: E8 first, then the E9 runs for the seeds E8 did not cover."""
    tail = f'a1_resnet34_b2_f{frac}_d*_s*'
    return {**load(str(Path(root) / 'e9' / f'e9_base_{tail}')), **load(str(Path(root) / 'e8' / f'e8_base_{tail}'))}


def paired_with_base(root, frac, weight):
    """Prediction frames on the tags both arms have, and their design; None until the arm has runs."""
    arm = load(str(Path(root) / 'e9' / f'e9_l{weight}_a1_resnet34_b2_f{frac}_d*_s*'))
    base = base_sources(root, frac)
    tags = sorted(set(arm) & set(base))
    if not tags:
        return None
    return [pd.read_csv(arm[t]) for t in tags], [pd.read_csv(base[t]) for t in tags], run_levels(tags)


def _mean(fn, frames, *args):
    return float(np.nanmean([fn(df, *args) for df in frames]))


def _subset_metrics(frames):
    rows = [compute_metrics(s['gt'], s['pred']) for s in map(monotone_subset, frames) if len(s)]
    return {k: float(np.nanmean([r[k] for r in rows])) if rows else float('nan') for k in ('bacc4', 'qwk')}


def analyse(root, n_boot=10000):
    root = Path(root)
    choice = json.loads((root / 'e9' / 'choice.json').read_text())
    primary = []
    for frac in FRACTIONS:
        pair = paired_with_base(root, frac, choice[frac]['weight'])
        if pair:
            arm, base, levels = pair
            primary.append({
                'frac': frac, 'weight': choice[frac]['weight'], **contrast(arm, base, levels, n_boot, icc=True),
                'auroc': {'arm': _mean(within_joint_auroc, arm), 'base': _mean(within_joint_auroc, base)},
                'decline': {'arm': _mean(decline_share, arm), 'base': _mean(decline_share, base),
                            'reference': _mean(decline_share, base, 'gt')},
                'monotone': {'arm': _subset_metrics(arm), 'base': _subset_metrics(base)},
                'predictions': {'arm': prediction_stats(arm), 'base': prediction_stats(base)},
            })
    family = [m for r in primary for m in (r['bacc4'], r['qwk'])]
    if family:
        for m, p in zip(family, holm(np.array([m['p'] for m in family]))):
            m['p_holm'] = float(p)
    for r in primary:
        r['verdict'] = verdict(r['bacc4'], r['qwk'], r['auroc']['arm'] - r['auroc']['base'],
                               r['decline']['arm'] - r['decline']['base'])
    return {'choice': choice, 'primary': primary}


def report(results, baseline=None):
    lines = ['# E9: penalty on a fall of the expected grade between visits (ResNet34 + A1 ImageNet + B2)', '',
             'Post hoc: the test set was already used, so these results cannot pick a configuration for the thesis. '
             'Rules fixed before the first run: `papers/e9_prespecification_2026-11-09.md`. '
             'Holm over BACC4 and QWK at both label fractions; ICC p-values are unadjusted.', '',
             '## Primary contrasts against seeded `base`', '',
             '| Labels | Weight | Runs | Metric | Base | Arm | Diff [95% CI] | Holm p |', '|---|---|---|---|---|---|---|---|']
    for r in results['primary']:
        for name, m in (('BACC4', r['bacc4']), ('QWK', r['qwk'])):
            lines.append(f"| {float(r['frac']):.0%} | {r['weight']} | {r['runs']} | {name} | {m['base']:.3f} | "
                         f"{m['variant']:.3f} | {m['diff']:+.3f} [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}] | {m['p_holm']:.3f} |")
    lines += ['', '## Verdict and checks', '',
              '| Labels | Verdict | Within-joint AUROC base / arm | Predicted fall base / arm | Reference fall | ICC diff | p |',
              '|---|---|---|---|---|---|---|']
    for r in results['primary']:
        a, d, i = r['auroc'], r['decline'], r['icc']
        lines.append(f"| {float(r['frac']):.0%} | **{r['verdict']}** | {a['base']:.3f} / {a['arm']:.3f} | "
                     f"{d['base']:.3f} / {d['arm']:.3f} | {d['reference']:.3f} | {i['diff']:+.3f} | {i['p']:.3f} |")
    lines += ['', '## Joints whose reference never falls (selected on the reference, descriptive)', '',
              '| Labels | BACC4 base / arm | QWK base / arm |', '|---|---|---|']
    for r in results['primary']:
        b, a = r['monotone']['base'], r['monotone']['arm']
        lines.append(f"| {float(r['frac']):.0%} | {b['bacc4']:.3f} / {a['bacc4']:.3f} | {b['qwk']:.3f} / {a['qwk']:.3f} |")
    lines += ['', '## Recall per grade', '', '| Labels | Arm | Grade 0 share, predicted | Recall 0 | 1 | 2 | 3 | 5 |',
              '|---|---|---|---|---|---|---|---|']
    for r in results['primary']:
        for arm in ('base', 'arm'):
            s = r['predictions'][arm]
            lines.append(f"| {float(r['frac']):.0%} | {arm} | {s['pred_share_grade0']:.3f} | " +
                         ' | '.join(f"{s['recall'][g]:.3f}" for g in ('0', '1', '2', '3', '5')) + ' |')
    if baseline:
        lines += ['', '## No-image baseline on test (joint type + logit adjustment, labels of that fraction)', '',
                  '| Labels | BACC4 | QWK |', '|---|---|---|']
        for frac in FRACTIONS:
            lines.append(f"| {float(frac):.0%} | {baseline[frac]['bacc4']:.3f} | {baseline[frac]['qwk']:.3f} |")
    lines += ['', '## Weight selection (validation only, d0 s42)', '',
              '| Labels | Chosen | Floor met | Base BACC4 / QWK | Cells (weight: BACC4 / QWK) |', '|---|---|---|---|---|']
    for frac in FRACTIONS:
        c = results['choice'][frac]
        if 'cells' in c:
            cells = '; '.join(f"{x['weight']}: {x['bacc4']:.3f} / {x['qwk']:.3f}" for x in c['cells'])
            lines.append(f"| {float(frac):.0%} | {c['weight']} | {c['floor_met']} | "
                         f"{c['base_val']['bacc4']:.3f} / {c['base_val']['qwk']:.3f} | {cells} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', nargs='?', default='report', choices=['report', 'choose'])
    p.add_argument('--runs', default='runs')
    p.add_argument('--base-root', help='Root that holds e8/; default --runs')
    p.add_argument('--out', default='runs/e9_analysis')
    p.add_argument('--n-boot', type=int, default=10000)
    args = p.parse_args(argv)
    if args.command == 'choose':
        choice = {frac: select_weight(args.runs, frac, args.base_root) for frac in FRACTIONS}
        (Path(args.runs) / 'e9' / 'choice.json').write_text(json.dumps(choice, indent=2))
        print(json.dumps({f: c['weight'] for f, c in choice.items()}))
        return
    results = analyse(args.runs, args.n_boot)
    baseline = no_image_baseline()
    text = report(results, baseline)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'report.md').write_text(text)
    (dest / 'e9.json').write_text(json.dumps({**results, 'baseline': baseline}, indent=2))
    print(text)


if __name__ == '__main__':
    main()
