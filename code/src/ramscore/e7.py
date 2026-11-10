"""E7: grade-0 undersampling on ResNet34 + head B0 at 100% labels, against the E2 B0 and E3 B2 runs of the same seeds.

  uperm, urot          grade 0 down to 1:1, dropped for good or a new subset every epoch; 50 epochs (8,250 steps)
  upermlong, urotlong  scratch only, 120 epochs (19,800 steps, as B0 and B2), to separate trimming from a smaller budget

Post hoc: the test set was already used. The contrasts, the Holm family and the verdict rules were fixed before the
first run in papers/e7_prespecification_2026-11-04.md. Paired patient bootstrap of compare.py (hier: seeds redrawn).

Usage: uv run --group analysis python -m ramscore.e7
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import NATIVE_SCORES
from .e6abl import add_holm, contrast, load, paired

ENCODERS = {'a0_resnet34': 'A0 scratch', 'a1_resnet34': 'A1 ImageNet'}
ARMS = ('uperm', 'urot')
LONG = {'upermlong': 'uperm', 'urotlong': 'urot'}
BASES = {'b0': ('e2', 'B0'), 'b2': ('e3', 'B2')}
MIN_DIFF = 0.05
ALPHA = 0.05
NAME = re.compile(r'^e7_(u[a-z]+)_(a\d_resnet34)_b0_f1\.0_d0_s(\d+)$')


def parse_name(name):
    m = NAME.match(name)
    return {'arm': m[1], 'enc': m[2], 'seed': int(m[3])} if m else None


def arm_glob(root, arm, enc):
    return str(Path(root) / 'e7' / f'e7_{arm}_{enc}_b0_f1.0_d*_s*')


def base_glob(root, base, enc):
    exp = BASES[base][0]
    return str(Path(root) / exp / f'{exp}_{enc}_{base}_f1.0_d*_s*')


def source_glob(root, name, enc):
    return base_glob(root, name, enc) if name in BASES else arm_glob(root, name, enc)


def verdict(base, stats):
    """Locked rules: B0 needs Holm p < .05 and at least +5 points; B2 needs Holm p < .05 and a CI above 0.
    'worse' is the same test in the other direction."""
    if stats['p_holm'] >= ALPHA:
        return 'not shown'
    if stats['diff'] < 0:
        return 'worse'
    better = stats['diff'] >= MIN_DIFF if base == 'b0' else stats['ci'][0] > 0
    return 'better' if better else 'not shown'


def prediction_stats(frames):
    """Recall per native grade and share of grade 0 among predictions and in truth, per run, then averaged."""
    recall = {str(g): [] for g in NATIVE_SCORES}
    pred, true = [], []
    for df in frames:
        pred.append(float((df['pred'] == 0).mean()))
        true.append(float((df['gt'] == 0).mean()))
        for g in NATIVE_SCORES:
            hit = df.loc[df['gt'] == g, 'pred'] == g
            recall[str(g)].append(float(hit.mean()) if len(hit) else float('nan'))
    return {'runs': len(frames), 'pred_share_grade0': float(np.mean(pred)), 'true_share_grade0': float(np.mean(true)),
            'recall': {g: float(np.nanmean(v)) for g, v in recall.items()}}


def analyse(root, n_boot=10000):
    root = Path(root)
    primary = []
    for enc in ENCODERS:
        for arm in ARMS:
            for base in BASES:
                pair = paired(arm_glob(root, arm, enc), base_glob(root, base, enc))
                if pair:
                    primary.append({'arm': arm, 'enc': enc, 'base': base, **contrast(*pair, n_boot, icc=True)})
    add_holm(primary)
    for r in primary:
        r['verdict'] = verdict(r['base'], r['bacc4'])

    plan = [('urot', enc, 'uperm') for enc in ENCODERS]
    plan += [(long, 'a0_resnet34', short) for long, short in LONG.items()]
    plan += [(long, 'a0_resnet34', base) for long in LONG for base in BASES]
    supporting = []
    for variant, enc, against in plan:
        pair = paired(arm_glob(root, variant, enc), source_glob(root, against, enc))
        if pair:
            supporting.append({'variant': variant, 'enc': enc, 'against': against, **contrast(*pair, n_boot, icc=True)})

    predictions = []
    for enc in ENCODERS:
        for name in (*BASES, *ARMS, *LONG):
            frames = [pd.read_csv(p) for p in load(source_glob(root, name, enc)).values()]
            if frames:
                predictions.append({'source': name, 'enc': enc, **prediction_stats(frames)})
    return {'primary': primary, 'supporting': supporting, 'predictions': predictions}


def cells(r):
    a, q, i = r['bacc4'], r['qwk'], r['icc']
    return (f"{r['runs']} | {a['base']:.3f} | {a['variant']:.3f} | {a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}]",
            f"{q['diff']:+.3f} | {q['p']:.3f} | {i['diff']:+.3f} | {i['p']:.3f} |")


def label(name):
    return BASES[name][1] if name in BASES else f'`{name}`'


def report(results):
    lines = ['# E7: grade-0 undersampling (ResNet34 + head B0, 100% labels)', '',
             'Post hoc: the test set was already used, so these results cannot pick a configuration for the thesis. '
             'Rules fixed before the first run: `papers/e7_prespecification_2026-11-04.md`. '
             'Holm over the 8 primary contrasts on BACC4; every other p-value is unadjusted.', '',
             '## Primary contrasts (50 epochs, 8,250 steps; B0 and B2 had 19,800)', '',
             '| Encoder | Arm | Against | Runs | BACC4 base | BACC4 arm | Diff [95% CI] | Holm p | Verdict | QWK diff | p | ICC diff | p |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results['primary']:
        left, right = cells(r)
        lines.append(f"| {ENCODERS[r['enc']]} | {label(r['arm'])} | {label(r['base'])} | {left} | "
                     f"{r['bacc4']['p_holm']:.3f} | {r['verdict']} | {right}")
    lines += ['', '## Supporting contrasts (outside the Holm family)', '',
              '| Encoder | Arm | Against | Runs | BACC4 base | BACC4 arm | Diff [95% CI] | p | QWK diff | p | ICC diff | p |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results['supporting']:
        left, right = cells(r)
        lines.append(f"| {ENCODERS[r['enc']]} | {label(r['variant'])} | {label(r['against'])} | {left} | "
                     f"{r['bacc4']['p']:.3f} | {right}")
    grades = [str(g) for g in NATIVE_SCORES]
    lines += ['', '## Recall per grade and over-calling', '',
              '| Encoder | Source | Runs | Grade 0 share, predicted | Grade 0 share, true | ' +
              ' | '.join(f'Recall {g}' for g in grades) + ' |', '|---|---|---|---|---|' + '---|' * len(grades)]
    for r in results['predictions']:
        lines.append(f"| {ENCODERS[r['enc']]} | {label(r['source'])} | {r['runs']} | {r['pred_share_grade0']:.3f} | "
                     f"{r['true_share_grade0']:.3f} | " + ' | '.join(f"{r['recall'][g]:.3f}" for g in grades) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='runs/e7_analysis')
    p.add_argument('--n-boot', type=int, default=10000)
    args = p.parse_args(argv)
    results = analyse(args.runs, args.n_boot)
    text = report(results)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'report.md').write_text(text)
    (dest / 'e7.json').write_text(json.dumps(results, indent=2))
    print(text)


if __name__ == '__main__':
    main()
