"""Paired patient-level bootstrap test of strategies against a baseline.

All runs predict the same test joints, so each replicate resamples the test
patients once and scores every run on that same sample. A strategy's score is
the mean over its seeds, and the difference to the baseline keeps the pairing.
Two intervals are reported per contrast:
  fixed  models held fixed, only test patients resampled (test-set uncertainty)
  hier   runs also redrawn in every replicate along the training design: patient
         draws with replacement, then seeds within each drawn draw. Strategies on
         the same design redraw the same runs, so run-level pairing is kept. This
         adds training variance but is coarse with 3 draws or 3 seeds, and the
         pre-trained encoder stays fixed, so intervals are conditional on it.
p-values are two-sided bootstrap p-values of the hier difference; Holm's
correction is applied over the contrasts against the baseline. Extra contrasts
(--pairs) are exploratory and left uncorrected.

Metrics come from per-patient confusion matrices, so a replicate is a weighted
sum of those matrices; they match metrics.compute_metrics (see tests).

Usage: python -m ramscore.compare b0 b1 b2 b3 b4 b5 b6 --pattern 'runs/{}_resnet34_f1.0_s*' \
    --pairs b2:b6 b4:b3 --out runs/compare_b_resnet34_f1.0
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LEVEL4, NATIVE_SCORES

KEYS = ('bacc4', 'bacc5', 'qwk', 'macro_f1', 'recall_ge2', 'binary_sensitivity', 'mae')
LOWER_IS_BETTER = {'mae'}
RUN_TAG = re.compile(r'(?:(?:^|_)d(?P<draw>\d+))?(?:(?:^|_)s(?P<seed>\d+))?$')


def run_levels(names):
    """(draw, seed) per run name; a missing draw is 0, and a run with neither tag is told apart by position."""
    levels = []
    for i, name in enumerate(names):
        tag = RUN_TAG.search(name)
        draw, seed = tag['draw'], tag['seed']
        levels.append((int(draw) if draw else 0, int(seed) if seed else (0 if draw else i)))
    return levels


def resample_indices(levels, n_boot, rng):
    """(n_boot, n_runs) run indices, resampled level by level along a balanced nested design.

    levels holds one tuple of nesting labels per run, outermost first, such as (draw, seed).
    """
    order = sorted(range(len(levels)), key=lambda i: levels[i])
    depth = len(levels[0])
    sizes = []
    for k in range(depth):
        children = {}
        for run in levels:
            children.setdefault(tuple(run[:k]), set()).add(run[k])
        counts = {len(v) for v in children.values()}
        if len(counts) != 1:
            raise ValueError(f'unbalanced run design at level {k}: {levels}')
        sizes.append(counts.pop())
    if int(np.prod(sizes)) != len(levels):
        raise ValueError(f'run labels are not unique: {levels}')
    table = np.array(order).reshape(sizes)
    picks = []
    for k, size in enumerate(sizes):
        drawn = rng.integers(0, size, size=(n_boot, *sizes[:k + 1]))
        picks.append(drawn.reshape(n_boot, *sizes[:k + 1], *[1] * (depth - k - 1)))
    return table[tuple(picks)].reshape(n_boot, -1)


class RunPicks:
    """Run indices per design: cells on one design redraw the same runs, also across separate analyses."""

    def __init__(self, n_boot, seed=0):
        self.n_boot, self.seed, self.memo = n_boot, seed, {}

    def __call__(self, levels):
        key = tuple(tuple(run) for run in levels)
        if key not in self.memo:
            self.memo[key] = resample_indices(key, self.n_boot, np.random.default_rng([self.seed, len(key)]))
        return self.memo[key]


def flat_levels(n_runs):
    """Design of a cell whose runs share one level: the runs alone are resampled."""
    return [(0, i) for i in range(n_runs)]


def patient_confusions(df, patients, score_values=NATIVE_SCORES):
    """(n_patients, K, K) confusion counts; df has 'patient', 'gt', 'pred'."""
    index = {s: i for i, s in enumerate(score_values)}
    k = len(score_values)
    p = pd.Index(patients).get_indexer(df['patient'])
    flat = (p * k + df['gt'].map(index).to_numpy()) * k + df['pred'].map(index).to_numpy()
    return np.bincount(flat, minlength=len(patients) * k * k).reshape(len(patients), k, k)


def _balanced_accuracy(cm):
    support = cm.sum(-1)
    recall = np.divide(np.diagonal(cm, axis1=-2, axis2=-1), support, out=np.zeros(support.shape), where=support > 0)
    return recall.sum(-1) / (support > 0).sum(-1)


def metrics_from_confusion(cm, score_values=NATIVE_SCORES):
    """Metrics of metrics.compute_metrics from confusion counts of shape (..., K, K)."""
    cm = np.asarray(cm, dtype=float)
    k = len(score_values)
    scores = np.asarray(score_values, dtype=float)
    n = cm.sum((-2, -1))
    levels = np.array([LEVEL4[s] for s in score_values])
    merge = np.eye(levels.max() + 1)[levels]                     # K x 4 one-hot
    cm4 = np.einsum('ki,...kl,lj->...ij', merge, cm, merge)
    pos = np.arange(k)
    weights = (pos[:, None] - pos[None, :]) ** 2
    expected = cm.sum(-1)[..., :, None] * cm.sum(-2)[..., None, :] / n[..., None, None]
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    f1_den = cm.sum(-1) + cm.sum(-2)
    ge2, gt0 = scores >= 2, scores > 0
    return {
        'bacc5': _balanced_accuracy(cm),
        'bacc4': _balanced_accuracy(cm4),
        'qwk': 1 - (weights * cm).sum((-2, -1)) / (weights * expected).sum((-2, -1)),
        'macro_f1': np.divide(2 * tp, f1_den, out=np.zeros(tp.shape), where=f1_den > 0).mean(-1),
        'recall_ge2': cm[..., ge2, :][..., ge2].sum((-2, -1)) / cm[..., ge2, :].sum((-2, -1)),
        'binary_sensitivity': cm[..., gt0, :][..., gt0].sum((-2, -1)) / cm[..., gt0, :].sum((-2, -1)),
        'mae': (np.abs(scores[:, None] - scores[None, :]) * cm).sum((-2, -1)) / n,
    }


def two_sided_p(draws):
    """Bootstrap p-value for H0: difference = 0, with the +1 correction."""
    b = len(draws)
    tail = min((draws <= 0).sum(), (draws >= 0).sum())
    return float(min(1.0, 2 * (tail + 1) / (b + 1)))


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def load_runs(strategies, pattern):
    """{strategy: [prediction DataFrame per seed run]} and the run names."""
    runs, names, ref = {}, {}, None
    for s in strategies:
        dirs = sorted(Path(p).parent for p in glob.glob(f'{pattern.format(s)}/test_predictions.csv'))
        if not dirs:
            raise SystemExit(f'no runs for {s}: {pattern.format(s)}')
        runs[s], names[s] = [], [d.name for d in dirs]
        for d in dirs:
            df = pd.read_csv(d / 'test_predictions.csv')
            key = df[['case', 'joint', 'gt', 'patient']]
            if ref is None:
                ref = key
            elif not key.equals(ref):
                raise SystemExit(f'{d} does not predict the same test joints as the first run')
            runs[s].append(df)
    return runs, names


def bootstrap(runs, n_boot=10000, seed=0, levels=None):
    """Point estimates, per-seed scores and paired bootstrap draws (fixed, hier) of every strategy.

    levels: (draw, seed) per run, aligned across strategies. A strategy with another number of runs, or any
    strategy when levels is None, has its runs resampled on one level.
    """
    patients = sorted(next(iter(runs.values()))[0]['patient'].unique())
    cms = {s: np.stack([patient_confusions(df, patients) for df in dfs]) for s, dfs in runs.items()}
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=n_boot)
    picks = RunPicks(n_boot, seed)
    point, fixed, hier, per_seed = {}, {}, {}, {}
    for s in runs:
        full = metrics_from_confusion(cms[s].sum(1))             # (seeds,) on the whole test set
        boot = metrics_from_confusion(np.einsum('bp,rpij->brij', w, cms[s]))   # (n_boot, seeds)
        n_runs = cms[s].shape[0]
        pick = picks(levels if levels is not None and len(levels) == n_runs else flat_levels(n_runs))
        per_seed[s] = {k: full[k].tolist() for k in KEYS}
        point[s] = {k: float(full[k].mean()) for k in KEYS}
        fixed[s] = {k: boot[k].mean(1) for k in KEYS}
        hier[s] = {k: np.take_along_axis(boot[k], pick, 1).mean(1) for k in KEYS}
    return point, fixed, hier, per_seed, patients


def aligned_levels(names):
    """Common (draw, seed) design of {strategy: run names}, using the longest arm; arms of that length must match."""
    designs = {s: run_levels(ns) for s, ns in names.items()}
    longest = max(designs.values(), key=len)
    for s, design in designs.items():
        if len(design) == len(longest) and design != longest:
            raise SystemExit(f'{s}: runs {design} do not match {longest}')
    return longest


def compare(runs, baseline, pairs=(), n_boot=10000, seed=0, alpha=0.05, levels=None):
    strategies = list(runs)
    point, fixed, hier, per_seed, patients = bootstrap(runs, n_boot, seed, levels)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def contrast(a, b):
        out = {}
        for k in KEYS:
            d_fixed, d_hier = fixed[a][k] - fixed[b][k], hier[a][k] - hier[b][k]
            out[k] = {
                'diff': point[a][k] - point[b][k],
                'ci_fixed': np.nanpercentile(d_fixed, [lo, hi]).tolist(),
                'ci_hier': np.nanpercentile(d_hier, [lo, hi]).tolist(),
                'p_fixed': two_sided_p(d_fixed[~np.isnan(d_fixed)]),
                'p_hier': two_sided_p(d_hier[~np.isnan(d_hier)]),
                'prob_better': float(np.mean(d_hier < 0 if k in LOWER_IS_BETTER else d_hier > 0)),
                'seed_pairs_better': int(sum((x < y) if k in LOWER_IS_BETTER else (x > y)
                                             for x, y in zip(per_seed[a][k], per_seed[b][k]))),
                'seed_pairs': min(len(per_seed[a][k]), len(per_seed[b][k])),
            }
        return out

    primary = {f'{s}-{baseline}': contrast(s, baseline) for s in strategies if s != baseline}
    for k in KEYS:
        for src, dst in (('p_hier', 'p_holm'), ('p_fixed', 'p_holm_fixed')):
            adj = holm([c[k][src] for c in primary.values()])
            for c, p in zip(primary.values(), adj):
                c[k][dst] = float(p)
    extra = {f'{a}-{b}': contrast(a, b) for a, b in pairs}
    return {
        'baseline': baseline, 'n_boot': n_boot, 'seed': seed, 'n_patients': len(patients),
        'n_joints': int(len(next(iter(runs.values()))[0])),
        'point': point, 'per_seed': per_seed,
        'ci_fixed': {s: {k: np.nanpercentile(fixed[s][k], [lo, hi]).tolist() for k in KEYS} for s in strategies},
        'vs_baseline': primary, 'pairs': extra,
    }


def to_markdown(res, keys=('bacc4', 'bacc5', 'qwk', 'recall_ge2')):
    def fmt_ci(ci):
        return f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'

    lines = [f"Paired patient bootstrap: {res['n_patients']} test patients, {res['n_joints']} joints, "
             f"{res['n_boot']} replicates; diff = strategy - {res['baseline']} (mean over seeds).", '']
    for k in keys:
        lines += [f'### {k}', '',
                  '| Contrast | Diff | 95% CI (fixed) | 95% CI (hier) | p (hier) | p Holm | P(better) | Seeds better |',
                  '|---|---|---|---|---|---|---|---|']
        for name, c in list(res['vs_baseline'].items()) + list(res['pairs'].items()):
            m = c[k]
            holm_p = f"{m['p_holm']:.3f}" if 'p_holm' in m else '—'
            lines.append(f"| {name} | {m['diff']:+.3f} | {fmt_ci(m['ci_fixed'])} | {fmt_ci(m['ci_hier'])} | "
                         f"{m['p_hier']:.3f} | {holm_p} | {m['prob_better']:.2f} | {m['seed_pairs_better']}/{m['seed_pairs']} |")
        lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('strategies', nargs='+', help='Strategy prefixes; the first is the baseline')
    p.add_argument('--pattern', default='runs/{}_resnet34_f1.0_s*', help='Run directory glob, {} = strategy')
    p.add_argument('--pairs', nargs='*', default=[], help='Extra contrasts a:b (exploratory, no correction)')
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', help='Write <out>.json and <out>.md')
    args = p.parse_args(argv)

    runs, names = load_runs(args.strategies, args.pattern)
    pairs = [tuple(x.split(':')) for x in args.pairs]
    for a, b in pairs:
        if a not in runs or b not in runs:
            raise SystemExit(f'pair {a}:{b} uses a strategy that is not loaded')
    res = compare(runs, args.strategies[0], pairs, n_boot=args.n_boot, seed=args.seed, levels=aligned_levels(names))
    res['runs'] = names
    md = to_markdown(res)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(f'{args.out}.json').write_text(json.dumps(res, indent=2))
        Path(f'{args.out}.md').write_text(md + '\n')


if __name__ == '__main__':
    main()
