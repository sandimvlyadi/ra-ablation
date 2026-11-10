"""E6 analysis: MAE ViT-S with the MAE fine-tuning recipe (layer-wise LR decay + warm-up) against E2.

Every E6 run uses an E2 patient draw and seed, so within a B x fraction cell the runs pair with
e2_a3_vits (MAE, shared recipe) and e2_a1_vits (ImageNet ViT-S, shared recipe); only draw/seed tags present
in every source are used. ImageNet and DINO ViT-S with the new recipe join automatically once their E6 runs
exist (fair comparison: MAE-LLRD vs ImageNet-LLRD).

  primary  10% labels, B0 and B4: MAE-LLRD - MAE-E2 and MAE-LLRD - ImageNet-E2 on BACC4 (Holm over the 4),
           paired patient bootstrap of compare.py (hier: runs redrawn); QWK and the hand-total ICC(A,1)
           (icc.py) alongside, with the same Holm family
  explore  100% labels and B6, and the LLRD-vs-LLRD contrasts when present; no correction

Usage: uv run --group analysis python -m ramscore.e6 --out runs/e6_analysis
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

SOURCES = {
    'mae_llrd': 'e6/e6_a3_vits', 'mae_e2': 'e2/e2_a3_vits', 'imagenet_e2': 'e2/e2_a1_vits',
    'imagenet_llrd': 'e6/e6_a1_vits', 'dino_llrd': 'e6/e6_a4_vits', 'dino_e2': 'e2/e2_a4_vits',
}
LABELS = {'mae_llrd': 'MAE, LLRD', 'mae_e2': 'MAE, E2 recipe', 'imagenet_e2': 'ImageNet, E2 recipe',
          'imagenet_llrd': 'ImageNet, LLRD', 'dino_llrd': 'DINO, LLRD', 'dino_e2': 'DINO, E2 recipe'}
PRIMARY = (('mae_llrd', 'mae_e2'), ('mae_llrd', 'imagenet_e2'))
EXPLORE = (('mae_llrd', 'imagenet_llrd'), ('mae_llrd', 'dino_llrd'), ('dino_llrd', 'dino_e2'),
           ('imagenet_llrd', 'imagenet_e2'))
CELLS = (('0.1', 'b0'), ('0.1', 'b4'), ('1.0', 'b0'), ('1.0', 'b4'), ('0.1', 'b6'), ('1.0', 'b6'))
KEYS = ('bacc4', 'qwk')
TAG = re.compile(r'_(d\d_s\d+)$')


def load_cell(root, frac, b):
    """{source: [predictions per run]} on the draw/seed tags that every source with runs shares."""
    found = {}
    for s, prefix in SOURCES.items():
        runs = {}
        for p in sorted(glob.glob(f'{root}/{prefix}_{b}_f{frac}_d*_s*/test_predictions.csv')):
            runs[TAG.search(Path(p).parent.name).group(1)] = p
        if runs:
            found[s] = runs
    if 'mae_llrd' not in found:
        return None
    tags = set.intersection(*(set(r) for r in found.values()))
    return {s: [pd.read_csv(r[t]) for t in sorted(tags)] for s, r in found.items()}, sorted(tags)


def analyse_cell(runs, n_boot, seed=0, alpha=0.05, levels=None):
    point, _, hier, _, _ = bootstrap(runs, n_boot, seed, levels)
    totals = {s: [hand_totals(df) for df in dfs] for s, dfs in runs.items()}
    ipoint, idraws, _ = bootstrap_cells(totals, n_boot, seed, levels=None if levels is None else {s: levels for s in runs})
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    summary = {s: {**{k: point[s][k] for k in KEYS}, 'icc': ipoint[s], 'icc_ci': ci(idraws[s])} for s in runs}
    rows = []
    for a, b in PRIMARY + EXPLORE:
        if a not in runs or b not in runs:
            continue
        row = {'contrast': f'{a} - {b}', 'primary': (a, b) in PRIMARY}
        for k in KEYS:
            d = hier[a][k] - hier[b][k]
            row[k] = {'diff': point[a][k] - point[b][k], 'ci': np.nanpercentile(d, [lo, hi]).tolist(),
                      'p': two_sided_p(d[~np.isnan(d)])}
        d = idraws[a] - idraws[b]
        row['icc'] = {'diff': ipoint[a] - ipoint[b], 'ci': ci(d), 'p': two_sided_p(d[~np.isnan(d)])}
        rows.append(row)
    return summary, rows


def analyse(root, n_boot=10000):
    results = {}
    for frac, b in CELLS:
        cell = load_cell(root, frac, b)
        if cell is None:
            continue
        runs, tags = cell
        summary, rows = analyse_cell(runs, n_boot, levels=run_levels(tags))
        results[(frac, b)] = {'tags': tags, 'summary': summary, 'contrasts': rows}
    family = [r for (frac, b), res in results.items() if frac == '0.1' and b in ('b0', 'b4')
              for r in res['contrasts'] if r['primary']]
    for k in (*KEYS, 'icc'):
        for r, p in zip(family, holm(np.array([r[k]['p'] for r in family])) if family else []):
            r[k]['p_holm'] = float(p)
    return results


def report(results):
    lines = ['# E6: MAE ViT-S with layer-wise LR decay + warm-up (vs E2 recipe)', '',
             'Paired on E2 patient draws and seeds; mean over runs; contrasts from the paired patient bootstrap '
             '(hier). Primary family: 10% labels, B0 and B4, MAE-LLRD vs MAE-E2 and vs ImageNet-E2 (Holm over 4).']
    for (frac, b), res in results.items():
        lines += ['', f"## {float(frac):.0%} labels, {b.upper()} ({len(res['tags'])} paired runs)", '',
                  '| Source | BACC4 | QWK | Hand ICC [95% CI] |', '|---|---|---|---|']
        for s, m in res['summary'].items():
            lines.append(f"| {LABELS[s]} | {m['bacc4']:.3f} | {m['qwk']:.3f} | {m['icc']:.3f} "
                         f"[{m['icc_ci'][0]:.3f}, {m['icc_ci'][1]:.3f}] |")
        if res['contrasts']:
            lines += ['', '| Contrast | BACC4 diff [95% CI] | p (Holm) | QWK diff | p (Holm) | ICC diff | p (Holm) |',
                      '|---|---|---|---|---|---|---|']
            for r in res['contrasts']:
                f = lambda k: f"{r[k]['p_holm']:.3f}" if 'p_holm' in r[k] else f"{r[k]['p']:.3f} (unadj.)"
                lines.append(f"| {r['contrast']} | {r['bacc4']['diff']:+.3f} [{r['bacc4']['ci'][0]:+.3f}, "
                             f"{r['bacc4']['ci'][1]:+.3f}] | {f('bacc4')} | {r['qwk']['diff']:+.3f} | {f('qwk')} | "
                             f"{r['icc']['diff']:+.3f} | {f('icc')} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='runs/e6_analysis')
    p.add_argument('--n-boot', type=int, default=10000)
    args = p.parse_args(argv)
    results = analyse(args.runs, args.n_boot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = report(results)
    tune = Path(args.runs) / 'e6' / 'tune_choice.txt'
    if tune.exists():
        text += '\n## Recipe tuning (10% d0 s42, B0, validation)\n\n```\n' + tune.read_text() + '```\n'
    (out / 'report.md').write_text(text)
    (out / 'e6.json').write_text(json.dumps({f'{f}_{b}': r for (f, b), r in results.items()}, indent=2))
    print(text)


if __name__ == '__main__':
    main()
