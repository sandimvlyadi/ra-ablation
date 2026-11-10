"""Secondary endpoint: agreement of the hand total erosion score between model and reference (E2-E4).

Hand total = sum of the native joint grades of one hand radiograph (a RAM-H1200 case: 16 joints; RheumaVIT:
the 13 shared joints). Agreement = ICC(2,1), two-way random effects, absolute agreement, single measure
(McGraw & Wong ICC(A,1)), between the reference totals and the model totals. The proposal (6) pre-registered
ICC(3,1) (consistency) of the patient total; ICC(A,1) of the hand total is used because it also penalises a
systematic offset (e.g. over-scoring on RheumaVIT), and ICC(C,1) = ICC(3,1) is reported beside it (cells.csv,
E4 table) for comparability with earlier studies.

CIs come from a patient bootstrap (all hands of a resampled patient go together; hands enter as frequency
weights), with the runs of every cell redrawn along the training design in each replicate ('hier', as compare.py); a cell's point
estimate is the mean ICC over its runs. Within one test set all cells share the replicates, so contrasts
are paired. p-values are two-sided bootstrap p-values, Holm within each family as in the primary analyses:

  E2  every encoder x B x fraction cell; contrast (i) at 10%: A3/A4 - A0/A1 with B0 and B4 (Holm over 12)
  E3  encoder x B at 10% and 100%; B4 - B1 at 100% per encoder (Holm over 3); H2 interaction
      (B4 - B1)_A4 - (B4 - B1)_A1 and _A0 at 100% (Holm over 2)
  E4  internal (RAM-H1200 test) and external (RheumaVIT test) ICC per configuration, drop = internal -
      external, drop(K2) - drop(K3) and drop(K4) - drop(K3) (Holm over 2); few-shot 0 / 5 / 20 patients

Usage: uv run --group analysis python -m ramscore.icc --out runs/icc_analysis
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import e2, e3, e4
from .compare import RunPicks, flat_levels, holm, run_levels, two_sided_p


def hand_totals(df, joints=None):
    """Per case: reference and predicted totals of the native grades, and the patient."""
    if joints is not None:
        df = df[df['joint'].isin(joints)]
    return df.groupby('case', sort=True).agg(patient=('patient', 'first'), gt=('gt', 'sum'), pred=('pred', 'sum'))


def icc_a1(x1, x2, w=None):
    """ICC(A,1) of two raters over n subjects; w = frequency weights of shape (n,) or (B, n) -> (B,)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    w = np.ones((1, len(x1))) if w is None else np.atleast_2d(w).astype(float)
    n = w.sum(1)
    r = (x1 + x2) / 2
    m = w @ r / n
    msr = 2 * (w @ r ** 2 - 2 * m * (w @ r) + m ** 2 * n) / (n - 1)
    e = (x1 - x2) / 2
    d = 2 * (w @ e) / n                                       # mean(x1) - mean(x2)
    mse = 2 * (w @ e ** 2 - d * (w @ e) + d ** 2 * n / 4) / (n - 1)
    msc = n * d ** 2 / 2
    return (msr - mse) / (msr + mse + 2 * (msc - mse) / n)


def icc_c1(x1, x2):
    """ICC(C,1) = ICC(3,1): two-way, consistency, single measure (ignores a constant offset)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    r, e = (x1 + x2) / 2, (x1 - x2) / 2
    n = len(x1)
    msr = 2 * ((r - r.mean()) ** 2).sum() / (n - 1)
    mse = 2 * ((e - e.mean()) ** 2).sum() / (n - 1)
    return (msr - mse) / (msr + mse)


def case_weights(patients_of_cases, n_boot, rng):
    """(n_boot, n_cases) bootstrap weights: patients drawn with replacement, each hand weighted by its patient."""
    patients, index = np.unique(patients_of_cases, return_inverse=True)
    w = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=n_boot)
    return w[:, index]


def bootstrap_cells(cells, n_boot=10000, seed=0, levels=None, pick_seed=0):
    """cells: {key: [hand_totals frame per run]} on one test set -> point ICC, hier draws, consistency ICC.

    A replicate is the mean ICC over runs redrawn along the training design. levels gives {key: nesting labels per
    run}; a key without labels has its runs resampled on one level. Cells on one design redraw the same runs, also
    across calls with the same pick_seed, so differences keep the run-level pairing.
    """
    rng = np.random.default_rng(seed)
    ref = next(iter(cells.values()))[0]
    w = case_weights(ref['patient'].to_numpy(), n_boot, rng)
    picks = RunPicks(n_boot, pick_seed)
    point, draws, per_run = {}, {}, {}
    consistency = {}
    for key, runs in cells.items():
        for t in runs:
            if not t.index.equals(ref.index) or not t['gt'].equals(ref['gt']):
                raise ValueError(f'{key}: runs do not score the same hands')
        boot = np.stack([icc_a1(t['gt'], t['pred'], w) for t in runs], 1)          # (n_boot, runs)
        full = np.array([icc_a1(t['gt'], t['pred'])[0] for t in runs])
        design = (levels or {}).get(key, flat_levels(len(runs)))
        point[key], per_run[key] = float(full.mean()), full.tolist()
        consistency[key] = float(np.mean([icc_c1(t['gt'], t['pred']) for t in runs]))
        draws[key] = np.take_along_axis(boot, picks(design), 1).mean(1)
    return point, draws, consistency


def ci(d, alpha=0.05):
    return np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)]).tolist()


def contrasts(family, point, draws, weights_list):
    """weights_list: [(label, {key: weight})]; Holm over the family."""
    rows = []
    for label, weights in weights_list:
        d = sum(c * draws[k] for k, c in weights.items())
        rows.append({'family': family, 'contrast': label, 'diff': sum(c * point[k] for k, c in weights.items()),
                     'ci': ci(d), 'p': two_sided_p(d[~np.isnan(d)])})
    for row, p in zip(rows, holm(np.array([r['p'] for r in rows]))):
        row['p_holm'] = float(p)
    return rows


def load_cells(pattern, joints=None):
    paths = sorted(glob.glob(f'{pattern}/test_predictions.csv'))
    return [hand_totals(pd.read_csv(p), joints) for p in paths]


def cell_levels(pattern):
    """(draw, seed) design of the runs that load_cells reads."""
    return run_levels([Path(p).parent.name for p in sorted(glob.glob(f'{pattern}/test_predictions.csv'))])


def analyse_e2(root, n_boot):
    cells, levels = {}, {}
    for bb, encoders in e2.ENCODERS.items():
        for a in encoders:
            for b in e2.STRATEGIES:
                for frac in e2.FRACTIONS:
                    pattern = f'{root}/e2/e2_{a}_{bb}_{b}_f{frac}_d*_s*'
                    runs = load_cells(pattern)
                    if runs:
                        cells[(bb, a, b, frac)] = runs
                        levels[(bb, a, b, frac)] = cell_levels(pattern)
    point, draws, cons = bootstrap_cells(cells, n_boot, levels=levels)
    frac = e2.PRIMARY['frac']
    weights = [(f'{bb} {b}: {e2.LABELS[s]} - {e2.LABELS[r]}', {(bb, s, b, frac): 1, (bb, r, b, frac): -1})
               for bb in e2.CORE for s in e2.SSL[bb] for r in e2.REFS for b in e2.PRIMARY['strategies']]
    return cons, point, draws, contrasts('E2 (i) at 10%', point, draws, weights)


def analyse_e3(root, n_boot):
    patterns = {(a, b, frac): f'{root}/{e3.run_dir(a, b)}_f{frac}_d*_s*'
                for a in e3.ENCODERS for b in e3.STRATEGIES for frac in e3.FRACTIONS}
    cells = {key: load_cells(pattern) for key, pattern in patterns.items()}
    levels = {key: cell_levels(pattern) for key, pattern in patterns.items()}
    point, draws, cons = bootstrap_cells(cells, n_boot, levels=levels)
    b4, b1 = e3.PRIMARY
    prim = [(f'{e3.LABELS[a]}: B4 - B1', {(a, b4, '1.0'): 1, (a, b1, '1.0'): -1}) for a in e3.ENCODERS]
    h2 = [(f'(B4 - B1) A4 - {r}', {('a4', b4, '1.0'): 1, ('a4', b1, '1.0'): -1, (r, b4, '1.0'): -1,
                                   (r, b1, '1.0'): 1}) for r in ('a1', 'a0')]
    return cons, point, draws, contrasts('E3 primary at 100%', point, draws, prim) + \
        contrasts('E3 H2 at 100%', point, draws, h2)


def analyse_e4(root, n_boot, manifest):
    test = set(pd.read_csv(manifest).query("e4_split == 'test'")['patient'])
    joints = e4.SHARED_JOINTS
    internal, external, levels_in, levels_ex = {}, {}, {}, {}
    for c in e4.CONFIGS:
        runs = e4.source_runs(c)
        internal[c] = [hand_totals(pd.read_csv(r / 'test_predictions.csv'), joints) for r in runs]
        levels_in[c] = levels_ex[(c, 'all', 0)] = run_levels([r.name for r in runs])
        direct = [pd.read_csv(Path(root) / 'e4' / 'direct' / r.name / 'predictions.csv') for r in runs]
        external[(c, 'all', 0)] = [hand_totals(df[df['patient'].isin(test)], joints) for df in direct]
        names = [r.name for r in e4.fewshot_sources(c)]
        external[(c, 'fs', 0)] = [t for r, t in zip(runs, external[(c, 'all', 0)]) if r.name in names]
        for k in e4.SIZES:
            external[(c, 'fs', k)] = [hand_totals(pd.read_csv(Path(root) / 'e4' / 'fewshot' / f'k{k}_d{d}' / n /
                                                              'predictions.csv'), joints)
                                      for d in e4.DRAWS for n in names]
            levels_ex[(c, 'fs', k)] = e4.fewshot_levels(len(external[(c, 'fs', k)]))
    # The same pick_seed makes a configuration redraw the same source runs internally and externally
    pi, di, ci_ = bootstrap_cells(internal, n_boot, seed=0, levels=levels_in)
    pe, de, ce = bootstrap_cells(external, n_boot, seed=1, levels=levels_ex)
    point = {('drop', c): pi[c] - pe[(c, 'all', 0)] for c in e4.CONFIGS}
    draws = {('drop', c): di[c] - de[(c, 'all', 0)] for c in e4.CONFIGS}
    weights = [(f'drop({a}) - drop({b})', {('drop', a): 1, ('drop', b): -1}) for a, b in e4.PRIMARY]
    return (ci_, ce), (pi, pe, point), (di, de, draws), contrasts('E4 H3 drop', point, draws, weights)


def fmt(point, draws, key):
    lo, hi = ci(draws[key])
    return f'{point[key]:.3f} [{lo:.3f}, {hi:.3f}]'


def fmt_rows(rows):
    return [f"| {r['family']} | {r['contrast']} | {r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | "
            f"{r['p']:.3f} | {r['p_holm']:.3f} |" for r in rows]


def report(res2, res3, res4):
    lines = ['# Hand total erosion score agreement: ICC(A,1) (secondary endpoint)', '',
             'Mean ICC over the runs of a cell; 95% CI from the patient bootstrap with runs redrawn (hier).']
    _, p2, d2, c2 = res2
    lines += ['', '## E2 (test, 10% and 100% labels)', '', '| Backbone | Encoder | B | 10% | 100% |', '|---|---|---|---|---|']
    for bb, encoders in e2.ENCODERS.items():
        for a in encoders:
            for b in e2.STRATEGIES:
                if (bb, a, b, '0.1') in p2:
                    lines.append(f"| {bb} | {e2.LABELS[a]} | {b.upper()} | {fmt(p2, d2, (bb, a, b, '0.1'))} | "
                                 f"{fmt(p2, d2, (bb, a, b, '1.0'))} |")
    _, p3, d3, c3 = res3
    lines += ['', '## E3 (ResNet34, test)', '', '| Encoder | B | 10% | 100% |', '|---|---|---|---|']
    for a in e3.ENCODERS:
        for b in e3.STRATEGIES:
            lines.append(f"| {e3.LABELS[a]} | {b.upper()} | {fmt(p3, d3, (a, b, '0.1'))} | {fmt(p3, d3, (a, b, '1.0'))} |")
    (ci_, ce), (pi, pe, pd_), (di, de, dd), c4 = res4
    lines += ['', '## E4 (13 shared joints; external = RheumaVIT test)', '',
              '| Config | Internal | External, direct | Drop | External ICC(3,1), direct | '
              'Few-shot 0 / 5 / 20 patients (3 source runs) |', '|---|---|---|---|---|---|']
    for c in e4.CONFIGS:
        fs = ' / '.join(f"{pe[(c, 'fs', k)]:.3f}" for k in (0, *e4.SIZES))
        lines.append(f"| {c} {e4.LABELS[c]} | {fmt(pi, di, c)} | {fmt(pe, de, (c, 'all', 0))} | "
                     f"{fmt(pd_, dd, ('drop', c))} | {ce[(c, 'all', 0)]:.3f} | {fs} |")
    lines += ['', '## Contrasts (ICC difference; Holm within each family)', '',
              '| Family | Contrast | Difference [95% CI] | p | p (Holm) |', '|---|---|---|---|---|']
    lines += fmt_rows(c2 + c3 + c4)
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--manifest', default='runs/e4/rheumavit/manifest.csv')
    p.add_argument('--out', default='runs/icc_analysis')
    p.add_argument('--n-boot', type=int, default=10000)
    args = p.parse_args(argv)
    res2 = analyse_e2(args.runs, args.n_boot)
    res3 = analyse_e3(args.runs, args.n_boot)
    res4 = analyse_e4(args.runs, args.n_boot, args.manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = report(res2, res3, res4)
    (out / 'report.md').write_text(text)
    row = lambda exp, cell, v, d, c: {'experiment': exp, 'cell': cell, 'icc_a1': v, 'ci': ci(d), 'icc_c1': c}
    rows = [row('e2', '_'.join(k), v, res2[2][k], res2[0][k]) for k, v in res2[1].items()]
    rows += [row('e3', '_'.join(k), v, res3[2][k], res3[0][k]) for k, v in res3[1].items()]
    ((ci_, ce), (pi, pe, _), (di, de, _), _) = res4
    rows += [row('e4_internal', k, v, di[k], ci_[k]) for k, v in pi.items()]
    rows += [row('e4_external', '_'.join(map(str, k)), v, de[k], ce[k]) for k, v in pe.items()]
    pd.DataFrame(rows).to_csv(out / 'cells.csv', index=False)
    (out / 'icc.json').write_text(json.dumps({'contrasts': res2[3] + res3[3] + res4[3]}, indent=2))
    print(text)


if __name__ == '__main__':
    main()
