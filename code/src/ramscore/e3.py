"""E3 analysis: imbalance/ordinal strategy x encoder (proposal RQ2 / H2, 5.3 and 6).

  table    test and validation metrics of every run -> runs.csv; mean +- SD per encoder x B x fraction -> report
  primary  pre-registered contrast (ii): B4 - B1 at 100% labels per encoder (A0, A1, A4 ResNet34); paired patient
           bootstrap of compare.py (hier CI: the seed runs are redrawn), Holm over the 3 encoders, minimum
           meaningful difference +5 points BACC4
  h2       interaction (B4 - B1)_A4 - (B4 - B1)_A1 and (B4 - B1)_A4 - (B4 - B1)_A0 at 100%, Holm over the 2
  explore  every B vs B0 per encoder x fraction (Holm within each family of 6, as compare.py), the pairs in PAIRS,
           and the primary and H2 contrasts at 10%; no further correction
  mixed    BACC4 and QWK ~ A * B * fraction + (1 | draw/seed) with e2.mixed_model (3 encoders x 7 B x 2 fractions)
  select   configuration for E4 by mean validation BACC4 (proposal: "best" is chosen on validation, never test)
  figure   mean +- SD per encoder x B (strategies.png)

B0, B4 and B6 come from E2 (same recipe, patient draws and seeds); B1, B2, B3 and B5 from runs/e3. Within one
fraction all 21 encoder x B cells share one set of bootstrap replicates, so every contrast is paired.

Usage: uv run --group analysis python -m ramscore.e3 --out runs/e3_analysis
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import KEYS, LOWER_IS_BETTER, aligned_levels, bootstrap, holm, load_runs, two_sided_p
from .e2 import MMD, fmt_p, mixed_model

NAME = re.compile(r'(?P<exp>e[23])_(?P<A>a\d)_(?P<backbone>resnet34)_(?P<B>b\d)_f(?P<frac>[\d.]+)'
                  r'_d(?P<draw>\d)_s(?P<seed>\d+)$')
ENCODERS = ('a0', 'a1', 'a4')
STRATEGIES = ('b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6')
FROM_E2 = ('b0', 'b4', 'b6')
FRACTIONS = ('0.1', '1.0')
PRIMARY = ('b4', 'b1')
PAIRS = (('b2', 'b1'), ('b2', 'b4'), ('b4', 'b3'), ('b6', 'b1'), ('b2', 'b6'))
METRICS = ('bacc4', 'bacc5', 'qwk', 'macro_f1', 'recall_ge2')
LABELS = {'a0': 'A0 scratch', 'a1': 'A1 ImageNet', 'a4': 'A4 DINO'}
B_LABELS = {'b0': 'ordinal', 'b1': 'ordinal + class weight + oversampling', 'b2': 'softmax + logit adjustment',
            'b3': 'CORN', 'b4': 'CORN + logit adjustment', 'b5': 'hierarchical', 'b6': 'classifier retraining'}


def parse_name(name):
    m = NAME.match(name)
    if m is None:
        return None
    d = m.groupdict()
    return {**d, 'draw': int(d['draw']), 'seed': int(d['seed'])}


def run_dir(a, b):
    """Run-name prefix relative to the runs folder: B0/B4/B6 from E2, the rest from E3."""
    exp = 'e2' if b in FROM_E2 else 'e3'
    return f'{exp}/{exp}_{a}_resnet34_{b}'


def collect(runs_root):
    """One row per E3 cell run: factors from the run name plus test and validation metrics."""
    rows = []
    for a in ENCODERS:
        for b in STRATEGIES:
            for frac in FRACTIONS:
                for path in sorted(glob.glob(f'{runs_root}/{run_dir(a, b)}_f{frac}_d*_s*/metrics.json')):
                    name = Path(path).parent.name
                    factors = parse_name(name)
                    if factors is None:
                        continue
                    m = json.loads(Path(path).read_text())
                    rows.append({'run': name, **factors, **{k: m['test'][k] for k in METRICS},
                                 **{f'val_{k}': m['val'][k] for k in METRICS}})
    df = pd.DataFrame(rows)
    df['encoder'] = df['A'] + '_' + df['backbone']
    return df


def linear_contrast(point, fixed, hier, per_seed, weights, alpha=0.05):
    """sum(c * cell) over {cell: c}; a pair is {a: 1, b: -1}, an interaction has four cells."""
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for k in KEYS:
        d_fixed = sum(c * fixed[g][k] for g, c in weights.items())
        d_hier = sum(c * hier[g][k] for g, c in weights.items())
        per = sum(c * np.asarray(per_seed[g][k]) for g, c in weights.items())   # runs matched by draw and seed
        better = per < 0 if k in LOWER_IS_BETTER else per > 0
        out[k] = {
            'diff': float(sum(c * point[g][k] for g, c in weights.items())),
            'ci_fixed': np.nanpercentile(d_fixed, [lo, hi]).tolist(),
            'ci_hier': np.nanpercentile(d_hier, [lo, hi]).tolist(),
            'p_fixed': two_sided_p(d_fixed[~np.isnan(d_fixed)]),
            'p_hier': two_sided_p(d_hier[~np.isnan(d_hier)]),
            'prob_better': float(np.mean(d_hier < 0 if k in LOWER_IS_BETTER else d_hier > 0)),
            'runs_better': int(better.sum()), 'runs': int(len(per)),
        }
    return out


def load_fraction(runs_root, frac):
    """{'a4_b1': [prediction frames]} for every encoder x B at one fraction, and their shared (draw, seed) design."""
    keys = {f'{a}_{b}': run_dir(a, b) for a in ENCODERS for b in STRATEGIES}
    runs, names = load_runs(list(keys.values()), f'{runs_root}/{{}}_f{frac}_d*_s*')
    return {g: runs[p] for g, p in keys.items()}, aligned_levels(names)


def all_contrasts(runs_root, n_boot=10000, seed=0):
    rows = []

    def add(family, frac, label, weights, stats):
        rows.append({'family': family, 'frac': frac, 'contrast': label, 'cells': json.dumps(weights),
                     **{f'{k}_{m}': v for k, s in stats.items() if k in METRICS for m, v in s.items()}})

    for frac in FRACTIONS:
        runs, levels = load_fraction(runs_root, frac)
        point, fixed, hier, per_seed, _ = bootstrap(runs, n_boot, seed, levels)

        def stats(weights):
            return linear_contrast(point, fixed, hier, per_seed, weights)

        t, r = PRIMARY
        for a in ENCODERS:
            w = {f'{a}_{t}': 1, f'{a}_{r}': -1}
            add('primary' if frac == '1.0' else 'primary@10%', frac, f'{a}: {t}-{r}', w, stats(w))
        for ref in ('a1', 'a0'):
            w = {f'a4_{t}': 1, f'a4_{r}': -1, f'{ref}_{t}': -1, f'{ref}_{r}': 1}
            add('h2' if frac == '1.0' else 'h2@10%', frac, f'({t}-{r}) a4 - {ref}', w, stats(w))
        for a in ENCODERS:
            for b in STRATEGIES[1:]:
                w = {f'{a}_{b}': 1, f'{a}_b0': -1}
                add(f'vs_b0 {a}', frac, f'{a}: {b}-b0', w, stats(w))
            for x, y in PAIRS:
                w = {f'{a}_{x}': 1, f'{a}_{y}': -1}
                add('pairs', frac, f'{a}: {x}-{y}', w, stats(w))
    df = pd.DataFrame(rows)
    for metric in METRICS:
        df[f'{metric}_p_holm'] = np.nan
        for (family, frac), idx in df.groupby(['family', 'frac']).groups.items():
            if family in ('primary', 'h2') or family.startswith('vs_b0'):
                df.loc[idx, f'{metric}_p_holm'] = holm(df.loc[idx, f'{metric}_p_hier'].to_numpy())
    return df


def contrast_table(rows, metric='bacc4', holm_column=True, mmd=True):
    head = ('| Fraction | Contrast | Diff | 95% CI (hier) | p | ' + ('p Holm | ' if holm_column else '')
            + 'Runs better |' + (' >= +5? |' if mmd else ''))
    lines = [head, '|' + '---|' * (head.count('|') - 1)]
    for r in rows.itertuples():
        lo, hi = getattr(r, f'{metric}_ci_hier')
        diff = getattr(r, f'{metric}_diff')
        cells = [f'{float(r.frac):.0%}', r.contrast, f'{diff:+.3f}', f'[{lo:+.3f}, {hi:+.3f}]',
                 fmt_p(getattr(r, f'{metric}_p_hier'))]
        if holm_column:
            p_holm = getattr(r, f'{metric}_p_holm')
            cells.append('—' if np.isnan(p_holm) else fmt_p(p_holm))
        cells.append(f"{getattr(r, f'{metric}_runs_better')}/{getattr(r, f'{metric}_runs')}")
        if mmd:
            # meaningful = point estimate reaches the MMD; "excluded" = the whole CI lies below it
            cells.append('yes' if diff >= MMD else ('excluded' if hi < MMD else 'not excluded'))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def summary_table(df, metric):
    """Markdown table: one row per B, one column per encoder x fraction, mean +- SD (n)."""
    cols = [(a, f) for f in FRACTIONS for a in ENCODERS]
    lines = ['| B | ' + ' | '.join(f'{LABELS[a].split()[1]} {float(f):.0%}' for a, f in cols) + ' |',
             '|---|' + '---|' * len(cols)]
    for b in STRATEGIES:
        cells = []
        for a, f in cols:
            v = df.loc[(df.A == a) & (df.B == b) & (df.frac == f), metric]
            cells.append(f'{v.mean():.3f} ± {v.std():.3f} ({len(v)})' if len(v) > 1 else '—')
        lines.append(f'| {b.upper()} {B_LABELS[b]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def selection(df, top=5):
    """Encoder x B ranked by mean validation BACC4 per fraction; test columns are shown, never used to choose."""
    g = (df.groupby(['frac', 'A', 'B'])[['val_bacc4', 'val_qwk', 'bacc4', 'qwk']].mean()
         .reset_index().sort_values(['frac', 'val_bacc4'], ascending=[True, False]))
    lines = ['| Fraction | Rank | Encoder | B | Val BACC4 | Val QWK | (test BACC4) | (test QWK) |',
             '|---|---|---|---|---|---|---|---|']
    best = {}
    for frac, part in g.groupby('frac', sort=False):
        best[frac] = part.iloc[0][['A', 'B']].to_dict()
        for rank, r in enumerate(part.head(top).itertuples(), 1):
            lines.append(f'| {float(frac):.0%} | {rank} | {LABELS[r.A]} | {r.B.upper()} | {r.val_bacc4:.3f} | '
                         f'{r.val_qwk:.3f} | {r.bacc4:.3f} | {r.qwk:.3f} |')
    return '\n'.join(lines), best


def plot_strategies(df, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'a0': '#7f7f7f', 'a1': '#1f77b4', 'a4': '#ff7f0e'}
    x = np.arange(len(STRATEGIES))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for i, metric in enumerate(('bacc4', 'qwk')):
        for j, frac in enumerate(FRACTIONS):
            ax = axes[i, j]
            for k, a in enumerate(ENCODERS):
                g = df[(df.A == a) & (df.frac == frac)].groupby('B')[metric].agg(['mean', 'std']).reindex(STRATEGIES)
                ax.errorbar(x + (k - 1) * 0.18, g['mean'], yerr=g['std'], fmt='o', ms=5, capsize=3,
                            color=colors[a], label=LABELS[a])
            if metric == 'bacc4':
                ax.axhline(0.25, color='k', lw=0.6, ls=':')
            ax.set_xticks(x, [b.upper() for b in STRATEGIES])
            ax.set_title(f'{float(frac):.0%} labels · test {metric.upper()}', fontsize=10)
            ax.grid(alpha=0.3, axis='y')
        axes[i, 0].set_ylabel(f'test {metric.upper()}')
    axes[0, 1].legend(fontsize=8, loc='lower right')
    fig.suptitle('E3 strategy x encoder, ResNet34 (mean ± SD over runs; 10%: 3 draws x 3 seeds, 100%: 3 seeds)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mixed_tables(results):
    lines = ['| Term | ' + ' | '.join(f'{m.upper()}: df, Wald chi2, p' for m in results) + ' |',
             '|---|' + '---|' * len(results)]
    terms = [t['term'] for t in next(iter(results.values()))['tests']]
    for term in terms:
        cells = []
        for r in results.values():
            t = next(t for t in r['tests'] if t['term'] == term)
            cells.append(f"{t['df']}, {t['chi2']:.1f}, {t['p']:.3g}")
        lines.append(f'| {term.replace(":", " x ")} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    for m, r in results.items():
        v = ', '.join(f'{k} {s ** 0.5:.4f}' for k, s in r['variance'].items())
        lines.append(f"- {m.upper()}: {r['n_runs']} runs, {r['n_subsets']} patient subsets, converged "
                     f"{r['converged']}; SD of {v}")
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs', help='Folder that holds e2/ and e3/')
    p.add_argument('--out', default='runs/e3_analysis')
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = collect(args.runs)
    expected = len(ENCODERS) * len(STRATEGIES) * (9 + 3)
    if len(runs) != expected:
        raise SystemExit(f'{len(runs)} runs found, expected {expected}')
    runs.to_csv(out / 'runs.csv', index=False)
    contrasts = all_contrasts(args.runs, args.n_boot, args.seed)
    contrasts.to_json(out / 'contrasts.json', orient='records', indent=2)
    mixed = {m: mixed_model(runs, 'resnet34', m) for m in ('bacc4', 'qwk')}
    (out / 'mixed.json').write_text(json.dumps(mixed, indent=2))
    plot_strategies(runs, out / 'strategies.png')
    sel, best = selection(runs)
    (out / 'selection.json').write_text(json.dumps(best, indent=2))

    fam = contrasts.family
    explo = contrasts[fam.str.startswith('vs_b0') | fam.eq('pairs') | fam.str.endswith('@10%')]
    md = [f'# E3 analysis ({len(runs)} runs, {args.n_boot} bootstrap replicates)', '',
          '## Primary contrast (ii): B4 - B1 at 100% labels, Holm over 3 encoders', '',
          contrast_table(contrasts[fam == 'primary']), '',
          '## Same contrast, QWK', '', contrast_table(contrasts[fam == 'primary'], 'qwk', mmd=False), '',
          f'">= +5?": yes = point estimate >= +{MMD:.2f}; excluded = whole 95% CI below +{MMD:.2f}. '
          '"Runs better": runs matched by draw and seed in which the first term wins.', '',
          '## H2 interaction: does the B4 - B1 gain grow with SSL? (100%, Holm over 2)', '',
          contrast_table(contrasts[fam == 'h2'], mmd=False), '',
          contrast_table(contrasts[fam == 'h2'], 'qwk', mmd=False), '',
          '## Mixed model: metric ~ A x B x fraction + (1 | draw/seed)', '', mixed_tables(mixed), '',
          '## Configuration for E4, chosen on validation BACC4', '', sel, '',
          '## Test BACC4, mean ± SD (n runs)', '', summary_table(runs, 'bacc4'), '',
          '## Test QWK, mean ± SD (n runs)', '', summary_table(runs, 'qwk'), '',
          '## Test recall of grades >= 2, mean ± SD (n runs)', '', summary_table(runs, 'recall_ge2'), '',
          '## Exploratory contrasts, BACC4 (Holm only within each "vs B0" family of 6)', '',
          contrast_table(explo), '',
          '## Exploratory contrasts, QWK', '', contrast_table(explo, 'qwk', mmd=False), '']
    (out / 'report.md').write_text('\n'.join(md))
    print('\n'.join(md[:29]))


if __name__ == '__main__':
    main()
