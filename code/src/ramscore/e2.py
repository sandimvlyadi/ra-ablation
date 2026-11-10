"""E2 analysis: label efficiency of the pre-trained encoders (proposal 5.4 and 6).

  table    test metrics of every run -> runs.csv; mean +- SD per encoder x B x fraction -> report
  primary  pre-registered contrast (i): A3/A4 vs A0 and A1 at 10% labels with B0 and B4; paired
           patient bootstrap of compare.py (hier CI: the 9 draw x seed runs are redrawn), Holm over
           the 12 contrasts, minimum meaningful difference +5 points BACC4
  curve    the same contrasts at every fraction, for B6, and A5 vs A4 / A3 vs A4 (exploratory, no Holm)
  mixed    BACC4 ~ A * B * fraction + (1 | draw/seed) per backbone on the 7 core encoders, sum coding,
           Wald chi-square per term (A5 runs only at 10% and 100%, so it stays out)
  figure   label-efficiency curves per backbone and B (curves_<metric>.png)

The draw index is the --fraction-seed, so a draw is the same patient subset for every encoder.

Usage: uv run --group analysis python -m ramscore.e2 --out runs/e2_analysis
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import aligned_levels, compare, holm, load_runs

NAME = re.compile(r'e2_(?P<A>a\d)_(?P<backbone>resnet34|vits)_(?P<B>b\d)_f(?P<frac>[\d.]+)_d(?P<draw>\d)_s(?P<seed>\d+)$')
FRACTIONS = ('0.01', '0.05', '0.1', '0.25', '1.0')
STRATEGIES = ('b0', 'b4', 'b6')
ENCODERS = {'resnet34': ('a0', 'a1', 'a4', 'a5'), 'vits': ('a0', 'a1', 'a3', 'a4', 'a5')}
CORE = {'resnet34': ('a0', 'a1', 'a4'), 'vits': ('a0', 'a1', 'a3', 'a4')}
SSL = {'resnet34': ('a4',), 'vits': ('a3', 'a4')}
REFS = ('a0', 'a1')
EXTRA = {'resnet34': (('a5', 'a4'), ('a5', 'a1')), 'vits': (('a5', 'a4'), ('a5', 'a1'), ('a3', 'a4'))}
PRIMARY = {'frac': '0.1', 'strategies': ('b0', 'b4')}
MMD = 0.05
METRICS = ('bacc4', 'bacc5', 'qwk', 'macro_f1', 'recall_ge2')
LABELS = {'a0': 'A0 scratch', 'a1': 'A1 ImageNet', 'a3': 'A3 MAE', 'a4': 'A4 DINO', 'a5': 'A5 DINO+ChronoCon'}


def parse_name(name):
    m = NAME.match(name)
    if m is None:
        return None
    d = m.groupdict()
    return {**d, 'draw': int(d['draw']), 'seed': int(d['seed'])}


def collect(root):
    """One row per finished run: factors from the run name plus its test metrics."""
    rows = []
    for path in sorted(glob.glob(f'{root}/e2_*/metrics.json')):
        name = Path(path).parent.name
        factors = parse_name(name)
        if factors is None:
            continue
        test = json.loads(Path(path).read_text())['test']
        rows.append({'run': name, **factors, **{k: test[k] for k in METRICS}})
    df = pd.DataFrame(rows)
    df['encoder'] = df['A'] + '_' + df['backbone']
    return df


def summary_table(df, metric):
    """Markdown table: one row per encoder x B, one column per fraction, mean +- SD (n)."""
    lines = [f'| Encoder | B | ' + ' | '.join(f'{float(f):.0%}' for f in FRACTIONS) + ' |',
             '|---|---|' + '---|' * len(FRACTIONS)]
    for backbone, encoders in ENCODERS.items():
        for a in encoders:
            for b in STRATEGIES:
                cells = []
                for f in FRACTIONS:
                    v = df.loc[(df.backbone == backbone) & (df.A == a) & (df.B == b) & (df.frac == f), metric]
                    cells.append(f'{v.mean():.3f} ± {v.std():.3f} ({len(v)})' if len(v) > 1 else '—')
                lines.append(f'| {a}_{backbone} | {b.upper()} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def encoder_contrasts(root, backbone, b, frac, n_boot, seed):
    """compare.py on the encoders of one backbone x B x fraction: every SSL encoder vs A0 and A1, plus extras."""
    present = [a for a in ENCODERS[backbone] if glob.glob(f'{root}/e2_{a}_{backbone}_{b}_f{frac}_d*_s*/test_predictions.csv')]
    runs, names = load_runs([f'{a}_{backbone}' for a in present], f'{root}/e2_{{}}_{b}_f{frac}_d*_s*')
    wanted = [(t, r) for t in SSL[backbone] for r in REFS] + [p for p in EXTRA[backbone] if p[0] in present]
    pairs = [(f'{t}_{backbone}', f'{r}_{backbone}') for t, r in wanted]
    res = compare(runs, f'a1_{backbone}', pairs, n_boot=n_boot, seed=seed, levels=aligned_levels(names))
    out = []
    for t, r in wanted:
        key = f'{t}_{backbone}-{r}_{backbone}'
        c = res['pairs'][key]
        out.append({'backbone': backbone, 'B': b, 'frac': frac, 'contrast': f'{t}-{r}', 'target': t, 'ref': r,
                    'n_runs': len(names[f'{t}_{backbone}']), 'n_runs_ref': len(names[f'{r}_{backbone}']),
                    **{f'{k}_{m}': c[k][m] for k in METRICS for m in ('diff', 'ci_hier', 'ci_fixed', 'p_hier')}})
    return out


def all_contrasts(root, n_boot=10000, seed=0):
    rows = []
    for backbone in ENCODERS:
        for frac in FRACTIONS:
            for b in STRATEGIES:
                rows += encoder_contrasts(root, backbone, b, frac, n_boot, seed)
    df = pd.DataFrame(rows)
    df['primary'] = ((df.frac == PRIMARY['frac']) & df.B.isin(PRIMARY['strategies'])
                     & df.target.isin(['a3', 'a4']) & df.ref.isin(REFS))
    df['p_holm'] = np.nan
    df.loc[df.primary, 'p_holm'] = holm(df.loc[df.primary, 'bacc4_p_hier'].to_numpy())
    return df


def fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def contrast_table(rows, metric='bacc4', holm_column=True):
    head = '| Backbone | B | Fraction | Contrast | Diff | 95% CI (hier) | p | ' + ('p Holm | ' if holm_column else '') + '>= +5? |'
    lines = [head, '|' + '---|' * (head.count('|') - 1)]
    for r in rows.itertuples():
        lo, hi = getattr(r, f'{metric}_ci_hier')
        diff = getattr(r, f'{metric}_diff')
        # meaningful = point estimate reaches the MMD; "excluded" = the whole CI lies below it
        verdict = 'yes' if diff >= MMD else ('excluded' if hi < MMD else 'not excluded')
        holm_cell = f' {fmt_p(r.p_holm)} |' if holm_column else ''
        lines.append(f'| {r.backbone} | {r.B.upper()} | {float(r.frac):.0%} | {LABELS[r.target].split()[0]} - '
                     f'{LABELS[r.ref].split()[0]} | {diff:+.3f} | [{lo:+.3f}, {hi:+.3f}] | '
                     f"{fmt_p(getattr(r, f'{metric}_p_hier'))} |{holm_cell} {verdict} |")
    return '\n'.join(lines)


def term_of(column):
    """'C(A, Sum)[S.a0]:C(B, Sum)[S.b0]' -> 'A:B'; 'Intercept' stays."""
    parts = [re.sub(r'^C\((\w+).*$', r'\1', p.split('[')[0]) for p in column.split(':')]
    return ':'.join(parts)


def mixed_model(df, backbone, metric='bacc4'):
    """BACC ~ A * B * fraction + (1 | draw/seed): random intercept per patient subset, seed within subset."""
    import statsmodels.formula.api as smf

    d = df[(df.backbone == backbone) & df.A.isin(CORE[backbone])].copy()
    d['subset'] = d['frac'] + '_d' + d['draw'].astype(str)
    model = smf.mixedlm(f'{metric} ~ C(A, Sum) * C(B, Sum) * C(frac, Sum)', d, groups='subset',
                        re_formula='1', vc_formula={'seed': '0 + C(seed)'})
    res = model.fit(reml=True, method='lbfgs', maxiter=5000)   # the default bfgs stops short on ViT-S
    names = list(res.params.index)
    terms = {}
    for i, col in enumerate(model.exog_names):
        if col != 'Intercept':
            terms.setdefault(term_of(col), []).append(i)
    tests = []
    for term, idx in terms.items():
        L = np.zeros((len(idx), len(names)))
        L[np.arange(len(idx)), idx] = 1
        w = res.wald_test(L, scalar=True, use_f=False)
        tests.append({'term': term.replace('frac', 'fraction'), 'df': len(idx),
                      'chi2': float(np.squeeze(w.statistic)), 'p': float(w.pvalue)})
    variance = {'subset (draw)': float(res.cov_re.iloc[0, 0]), 'seed within subset': float(res.vcomp[0]),
                'residual': float(res.scale)}
    return {'backbone': backbone, 'n_runs': len(d), 'n_subsets': d['subset'].nunique(),
            'converged': bool(res.converged), 'tests': tests, 'variance': variance}


def mixed_table(results):
    lines = ['| Backbone | Term | df | Wald chi2 | p |', '|---|---|---|---|---|']
    for r in results:
        for t in r['tests']:
            lines.append(f"| {r['backbone']} | {t['term']} | {t['df']} | {t['chi2']:.1f} | {t['p']:.3g} |")
    lines.append('')
    for r in results:
        v = ', '.join(f'{k} {s ** 0.5:.4f}' for k, s in r['variance'].items())
        lines.append(f"- {r['backbone']}: {r['n_runs']} runs, {r['n_subsets']} patient subsets, "
                     f"converged {r['converged']}; SD of {v}")
    return '\n'.join(lines)


def plot_curves(df, metric, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'a0': '#7f7f7f', 'a1': '#1f77b4', 'a3': '#2ca02c', 'a4': '#ff7f0e', 'a5': '#9467bd'}
    x = {f: float(f) * 100 for f in FRACTIONS}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for i, (backbone, encoders) in enumerate(ENCODERS.items()):
        for j, b in enumerate(STRATEGIES):
            ax = axes[i, j]
            for a in encoders:
                g = df[(df.backbone == backbone) & (df.A == a) & (df.B == b)].groupby('frac')[metric]
                stats = g.agg(['mean', 'std']).reindex([f for f in FRACTIONS if f in g.groups])
                ax.errorbar([x[f] for f in stats.index], stats['mean'], yerr=stats['std'], color=colors[a],
                            marker='o', ms=4, capsize=3, lw=1.5, ls='--' if a == 'a5' else '-', label=LABELS[a])
            if metric.startswith('bacc'):
                ax.axhline(0.25 if metric == 'bacc4' else 0.2, color='k', lw=0.6, ls=':')
            ax.set_xscale('log')
            ax.set_xticks([1, 5, 10, 25, 100], ['1', '5', '10', '25', '100'])
            ax.set_title(f"{'ResNet34' if backbone == 'resnet34' else 'ViT-S/16'} · {b.upper()}", fontsize=10)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(f'test {metric.upper()}')
            if i == 1:
                ax.set_xlabel('label fraction (% of 154 train patients)')
        axes[i, -1].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'E2 label efficiency, test {metric.upper()} (mean ± SD over draws x seeds)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='runs/e2')
    p.add_argument('--out', default='runs/e2_analysis')
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = collect(args.root)
    runs.to_csv(out / 'runs.csv', index=False)
    contrasts = all_contrasts(args.root, args.n_boot, args.seed)
    contrasts.to_json(out / 'contrasts.json', orient='records', indent=2)
    mixed = [mixed_model(runs, b) for b in ENCODERS]
    (out / 'mixed.json').write_text(json.dumps(mixed, indent=2))
    for metric in ('bacc4', 'qwk'):
        plot_curves(runs, metric, out / f'curves_{metric}.png')

    explo = contrasts[~contrasts.primary]
    md = [f'# E2 analysis ({len(runs)} runs, {args.n_boot} bootstrap replicates)', '',
          '## Primary contrast (i): 10% labels, B0 and B4, Holm over 12', '',
          contrast_table(contrasts[contrasts.primary]), '',
          f'">= +5?": yes = point estimate >= +{MMD:.2f}; excluded = whole 95% CI below +{MMD:.2f}.', '',
          '## Mixed model: BACC4 ~ A x B x fraction + (1 | draw/seed)', '', mixed_table(mixed), '',
          '## Test BACC4, mean ± SD (n runs)', '', summary_table(runs, 'bacc4'), '',
          '## Test QWK, mean ± SD (n runs)', '', summary_table(runs, 'qwk'), '',
          '## Exploratory contrasts, BACC4 (no multiplicity correction)', '',
          contrast_table(explo, holm_column=False), '',
          '## Exploratory contrasts, QWK (no multiplicity correction)', '',
          contrast_table(contrasts, 'qwk', holm_column=False), '']
    (out / 'report.md').write_text('\n'.join(md))
    print('\n'.join(md[:8 + 2]))
    print(mixed_table(mixed))


if __name__ == '__main__':
    main()
