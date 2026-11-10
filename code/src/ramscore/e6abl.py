"""E6 ablations on ResNet34 + B2, paired with the E3 and E2 runs of the same patient draws and seeds.

  mask    bone-mask input channel (seg.py) vs image only: A0, A1, A4 at 10% and 100%. BACC4, QWK, hand ICC.
  levels  3 classes {0, 1, >= 2} in training vs 5 native grades merged after prediction, scored on 3 levels (BACC3,
          QWK): A0, A1, A4 at 10% and 100% (proposal 4, sensitivity analysis).
  pool    DINO pre-trained on RAM-H1200 + RAM-W600 vs RAM-H1200 only: A4 at 10% and 100%. Primary analysis of
          RAM-H1200 keeps W600 out because its patients cannot be matched to the RAM-H1200 test patients, so a
          gain here would not be free of possible leakage.

All are exploratory. Paired patient bootstrap of compare.py (hier: runs redrawn), Holm over the cells of one ablation
on the first metric. Minimum meaningful difference +5 points, as in the proposal.

Usage: uv run --group analysis python -m ramscore.e6abl --out runs/e6abl/analysis
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import bootstrap, holm, run_levels, two_sided_p
from .icc import bootstrap_cells, ci, hand_totals

ENCODERS = {'a0_resnet34': 'A0 scratch', 'a1_resnet34': 'A1 ImageNet', 'a4_resnet34': 'A4 DINO'}
FRACTIONS = ('0.1', '1.0')
TAG = re.compile(r'_(d\d_s\d+)$')
KEYS = ('bacc4', 'qwk')


def load(pattern):
    return {TAG.search(Path(p).parent.name).group(1): p for p in sorted(glob.glob(f'{pattern}/test_predictions.csv'))}


def collapse(df):
    """Native grades -> {0, 1, >= 2}."""
    return df.assign(gt=df['gt'].clip(upper=2), pred=df['pred'].clip(upper=2))


def contrast(variant, base, levels, n_boot, seed=0, icc=False, alpha=0.05):
    """variant and base: lists of paired prediction frames on one (draw, seed) design. Returns point values and
    the difference per metric."""
    runs = {'variant': variant, 'base': base}
    point, _, hier, _, _ = bootstrap(runs, n_boot, seed, levels)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    row = {'runs': len(variant)}
    for k in KEYS:
        d = hier['variant'][k] - hier['base'][k]
        row[k] = {'variant': point['variant'][k], 'base': point['base'][k], 'diff': point['variant'][k] - point['base'][k],
                  'ci': np.nanpercentile(d, [lo, hi]).tolist(), 'p': two_sided_p(d[~np.isnan(d)])}
    if icc:
        totals = {s: [hand_totals(df) for df in dfs] for s, dfs in runs.items()}
        ipoint, idraws, _ = bootstrap_cells(totals, n_boot, seed, levels={s: levels for s in runs})
        d = idraws['variant'] - idraws['base']
        row['icc'] = {'variant': ipoint['variant'], 'base': ipoint['base'], 'diff': ipoint['variant'] - ipoint['base'],
                      'ci': ci(d), 'p': two_sided_p(d[~np.isnan(d)])}
    return row


def paired(variant_glob, base_glob, transform=None):
    """Prediction frames on the tags both sources have, and their (draw, seed) design; None until the variant has runs."""
    v, b = load(variant_glob), load(base_glob)
    tags = sorted(set(v) & set(b))
    if not tags:
        return None
    f = transform or (lambda df: df)
    return [pd.read_csv(v[t]) for t in tags], [f(pd.read_csv(b[t])) for t in tags], run_levels(tags)


def add_holm(rows):
    if rows:
        for r, p in zip(rows, holm(np.array([r['bacc4']['p'] for r in rows]))):
            r['bacc4']['p_holm'] = float(p)
    return rows


def analyse(root, n_boot=10000):
    root = Path(root)
    out = {'mask': [], 'levels': [], 'pool': []}
    for frac in FRACTIONS:
        for enc in ENCODERS:
            tail = f'{enc}_b2_f{frac}_d*_s*'
            base = str(root / 'e3' / f'e3_{tail}')
            pair = paired(str(root / 'e6abl' / f'e6abl_mask_{tail}'), base)
            if pair:
                out['mask'].append({'frac': frac, 'enc': enc, **contrast(*pair, n_boot, icc=True)})
            pair = paired(str(root / 'e6abl' / f'e6abl_lv3_{tail}'), base, collapse)
            if pair:
                out['levels'].append({'frac': frac, 'enc': enc, **contrast(*pair, n_boot)})
        tail = f'a4_resnet34_b2_f{frac}_d*_s*'
        pair = paired(str(root / 'e6abl' / f'e6abl_pool_{tail}'), str(root / 'e3' / f'e3_{tail}'))
        if pair:
            out['pool'].append({'frac': frac, 'enc': 'a4_resnet34', **contrast(*pair, n_boot, icc=True)})
    return {k: add_holm(v) for k, v in out.items()}


def table(rows, first, icc):
    head = f'| Labels | Encoder | Runs | {first} base | {first} variant | Diff [95% CI] | Holm p | QWK base | QWK variant | QWK diff | p |'
    sep = '|---|---|---|---|---|---|---|---|---|---|---|'
    if icc:
        head += ' ICC base | ICC variant | ICC diff | p |'
        sep += '---|---|---|---|'
    lines = [head, sep]
    for r in rows:
        a, q = r['bacc4'], r['qwk']
        line = (f"| {float(r['frac']):.0%} | {ENCODERS[r['enc']]} | {r['runs']} | {a['base']:.3f} | {a['variant']:.3f} | "
                f"{a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {a['p_holm']:.3f} | {q['base']:.3f} | "
                f"{q['variant']:.3f} | {q['diff']:+.3f} | {q['p']:.3f} |")
        if icc:
            i = r['icc']
            line += f" {i['base']:.3f} | {i['variant']:.3f} | {i['diff']:+.3f} | {i['p']:.3f} |"
        lines.append(line)
    return lines


def report(results, seg=None):
    lines = ['# E6 ablations (ResNet34 + B2, paired with E3)', '',
             'Exploratory. Holm over the cells of one ablation on the first metric; the other p-values are unadjusted.']
    if seg:
        lines += ['', f"Segmentation: test Dice {seg['test_dice']:.3f}, validation Dice {seg['val_dice']:.3f} (epoch {seg['epoch']})."]
    for key, title, first, icc in (('mask', 'Bone-mask channel vs image only', 'BACC4', True),
                                   ('levels', '3 classes in training vs 5 grades merged afterwards', 'BACC3', False),
                                   ('pool', 'Pre-training pool with RAM-W600 vs RAM-H1200 only', 'BACC4', True)):
        if results[key]:
            lines += ['', f'## {title}', '', *table(results[key], first, icc)]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='runs/e6abl/analysis')
    p.add_argument('--n-boot', type=int, default=10000)
    args = p.parse_args(argv)
    results = analyse(args.runs, args.n_boot)
    seg = Path(args.runs) / 'e6abl' / 'seg' / 'metrics.json'
    text = report(results, json.loads(seg.read_text()) if seg.exists() else None)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'report.md').write_text(text)
    (dest / 'e6abl.json').write_text(json.dumps(results, indent=2))
    print(text)


if __name__ == '__main__':
    main()
