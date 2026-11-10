"""E5: calibration, ordinal conformal prediction sets and referral (proposal RQ4 / H4, 5.5).

No training: the 18 source runs of the 4 E4 configurations (K1 9 runs, K2-K4 3 runs each; all B2 softmax,
ResNet34) are evaluated from their checkpoints. They were chosen on validation before E5, so E5 does not depend
on any test result of its own.

  logits   every run predicts RAM-H1200 val + test and all RheumaVIT joints -> runs/e5/logits/<run>.npz, and the
           argmax must reproduce the run's stored test and direct-transfer predictions, up to near-ties flipped
           by float rounding (top-2 logit margin < 0.01).
  analyse  on the 4 levels (0, 1, 2, >= 3), fixed before any E5 result:

  1. Temperature T on the calibration set (NLL). ECE per level before and after, on test.
  2. Contiguous ordinal sets (ordinal APS, Lu et al., 2022): start at the most probable level, add the more
     probable neighbour until the summed probability exceeds lambda. The score of a joint is the smallest lambda
     whose set holds its true level, i.e. the mass already in the set when that level joins; lambda = the
     conformal quantile of these scores (rank ceil((n + 1)(1 - alpha)), alpha = 0.10).
  3. Referral: a joint goes to a reader if its set holds more than one level or holds a level >= 2.

  Scopes   internal: calibrate on RAM-H1200 val (36 patients), test on RAM-H1200 test (101 patients, 16 joints)
           external: same calibration, RheumaVIT test (90 patients, 13 joints), reference = lower median
           local:    (exploratory) direct-transfer model, T and lambda refitted on 5 or 20 labelled RheumaVIT
                     pool patients (the 3 nested E4 draws), RheumaVIT test
  Variant  'prior' (secondary): B2 trains with balanced softmax, so its raw softmax targets balanced classes.
           Adding the log training prior back to the logits restores the posterior (Menon et al., 2021).
           Primary = the raw logits, as pre-registered.

  H4 criteria, internal scope, primary variant, per configuration:
     (a) coverage: the 95% CI of the marginal coverage contains 0.90 or lies above it
     (b) concentration: the 95% CI of the enrichment (grade >= 2 share among referred / overall) lies above 1
  Reported alongside, because they decide clinical use: referral rate, share of grade >= 2 joints referred, share
  of grade >= 2 errors referred, coverage per level. By the rule, a non-referred joint has a set {0} or {1}, so the
  planned "recall grade >= 2 among non-referred" is 0 by construction; the share of grade >= 2 among non-referred
  joints and the share of grade >= 2 joints referred carry that information.

  CI: 95% percentile, 2,000 replicates. Calibration and test patients are both resampled and lambda is recomputed
  per replicate, so calibration uncertainty is included; runs are redrawn (hier). T stays fixed within a replicate.
  Joints within a patient are dependent, so the conformal guarantee is only approximate.

Usage: uv run python -m ramscore.e5 logits && uv run python -m ramscore.e5 analyse
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from torch.utils.data import DataLoader

from . import rheumavit
from .compare import flat_levels, resample_indices, run_levels
from .data import LEVEL4, NATIVE_SCORES, BEDataset, load_records, subsample_patients
from .e4 import ALL_SCORES, CONFIGS, DRAWS, LABELS, SHARED_JOINTS, SIZES, fewshot_patients, source_runs
from .models import build_model, head_priors

ALPHA = 0.10
LEVELS = (0, 1, 2, 3)
METRICS = ('coverage', 'set_size', 'referral', 'sens_ge2', 'prev_ge2', 'prev_referred', 'prev_not_referred',
           'enrichment', 'high_error_flagged', 'accuracy_not_referred',
           'coverage_l0', 'coverage_l1', 'coverage_l2', 'coverage_l3')
IDX = {m: i for i, m in enumerate(METRICS)}


def level_probs(logits, temperature=1.0, offset=None):
    """Softmax over the 5 native grades, then grades 3 and 5 merged: N x 4 probabilities of the levels."""
    z = logits + offset if offset is not None else logits
    p = softmax(z / temperature, axis=1)
    return np.column_stack([p[:, 0], p[:, 1], p[:, 2], p[:, 3] + p[:, 4]])


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def fit_temperature(logits, y, offset=None):
    """T minimising the level NLL; searched on log T, so T stays positive."""
    res = minimize_scalar(lambda t: nll(level_probs(logits, math.exp(t), offset), y), bounds=(-3, 3), method='bounded',
                          options={'xatol': 1e-4})
    return math.exp(res.x)


def ece(prob, hit, bins=15):
    """Expected calibration error of one-vs-rest probabilities (equal-width bins)."""
    which = np.minimum((prob * bins).astype(int), bins - 1)
    total = np.bincount(which, minlength=bins)
    gap = np.abs(np.bincount(which, prob, bins) - np.bincount(which, hit.astype(float), bins))
    return float(gap.sum() / max(total.sum(), 1))


def expansion_mass(p):
    """N x K: probability mass already in the set when each level joins (0 for the mode).

    A level is in the set for lambda exactly when this mass is at most lambda; ties go to the lower grade.
    """
    prev = np.zeros_like(p)
    k = p.shape[1]
    for i, row in enumerate(p):
        lo = hi = int(row.argmax())
        mass = row[lo]
        while lo > 0 or hi < k - 1:
            left = row[lo - 1] if lo > 0 else -1.0
            right = row[hi + 1] if hi < k - 1 else -1.0
            if right > left:
                hi += 1
                prev[i, hi] = mass
                mass += row[hi]
            else:
                lo -= 1
                prev[i, lo] = mass
                mass += row[lo]
    return prev


def conformal_lambda(scores, alpha=ALPHA, weights=None):
    """Conformal quantile: the score of rank ceil((n + 1)(1 - alpha)); inf (the full set) if that exceeds n."""
    order = np.argsort(scores)
    w = np.ones(len(scores)) if weights is None else weights[order]
    cum = np.cumsum(w)
    rank = math.ceil((cum[-1] + 1) * (1 - alpha) - 1e-9)
    if rank > cum[-1]:
        return np.inf
    return float(scores[order[np.searchsorted(cum, rank - 1e-9)]])


def tally(member, y, pred, w=None):
    """The METRICS for boolean set membership (N x 4), levels y and point predictions; w = joint weights."""
    w = np.ones(len(y)) if w is None else w
    size = member.sum(1)
    referred = (size > 1) | member[:, 2:].any(1)
    covered = member[np.arange(len(y)), y]
    ge2 = y >= 2

    def share(event, among):
        den = w[among].sum()
        return w[among & event].sum() / den if den > 0 else np.nan

    everyone = np.ones(len(y), dtype=bool)
    prev_ge2 = share(ge2, everyone)
    prev_ref = share(ge2, referred)
    return np.array([
        share(covered, everyone), (w * size).sum() / w.sum(), share(referred, everyone), share(referred, ge2),
        prev_ge2, prev_ref, share(ge2, ~referred), prev_ref / prev_ge2 if prev_ge2 > 0 else np.nan,
        share(referred, ge2 & (pred < 2)), share(pred == y, ~referred),
        *(share(covered, y == level) for level in LEVELS)])


@torch.no_grad()
def collect(net, loader, device):
    net.eval()
    logits, gt, patient, joint, case = [], [], [], [], []
    for batch in loader:
        logits.append(net(batch['img'].to(device, non_blocking=True)).float().cpu().numpy())
        gt.append(batch['score'].numpy())
        patient += list(batch['patient'])
        joint += list(batch['joint'])
        case += list(batch['case'])
    return {'logits': np.concatenate(logits), 'gt': np.concatenate(gt), 'patient': np.array(patient),
            'joint': np.array(joint), 'case': np.array(case)}


def native_pred(logits):
    return np.array(NATIVE_SCORES)[logits.argmax(1)]


def check_reproduced(logits, stored, name, tie=1e-2):
    """Stored predictions must match the argmax, except near-ties flipped by float rounding (other batch size).

    Returns the number of such near-ties; anything else stops the extraction.
    """
    differ = native_pred(logits) != stored
    top = np.sort(logits[differ], axis=1)
    margin = top[:, -1] - top[:, -2]
    if (margin >= tie).any():
        raise SystemExit(f'{name}: {differ.sum()} predictions differ, largest top-2 margin {margin.max():.4f}')
    return int(differ.sum())


def extract(manifest, e4_out, out, batch_size=128, workers=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dest = Path(out) / 'logits'
    dest.mkdir(parents=True, exist_ok=True)
    runs = [r for c in CONFIGS for r in source_runs(c)]
    todo = [r for r in runs if not (dest / f'{r.name}.npz').exists()]
    if not todo:
        return
    root = json.loads((todo[0] / 'args.json').read_text())['data_root']
    loader = lambda ds: DataLoader(ds, batch_size=batch_size, num_workers=workers)
    loaders = {'val': loader(BEDataset(load_records(root, 'val'))), 'test': loader(BEDataset(load_records(root, 'test'))),
               'ext': loader(BEDataset(rheumavit.load_records(manifest), ALL_SCORES))}
    train = load_records(root, 'train')
    for run in todo:
        args = json.loads((run / 'args.json').read_text())
        if args['head'] != 'softmax':
            raise SystemExit(f'{run}: E5 expects B2 softmax runs')
        net = build_model(args['model'], len(NATIVE_SCORES), args['head'])
        net.load_state_dict(torch.load(run / 'model_best.pt', map_location='cpu')['model'])
        net.to(device)
        parts = {s: collect(net, lo, device) for s, lo in loaders.items()}
        stored = pd.read_csv(run / 'test_predictions.csv')
        direct = pd.read_csv(Path(e4_out) / 'direct' / run.name / 'predictions.csv')
        ties = check_reproduced(parts['test']['logits'], stored['pred'].to_numpy(), f'{run.name} test')
        ties += check_reproduced(parts['ext']['logits'], direct['pred'].to_numpy(), f'{run.name} external')
        records = subsample_patients(train, args['label_fraction'], args['fraction_seed'])
        to_class = {s: i for i, s in enumerate(NATIVE_SCORES)}
        prior = head_priors('softmax', [to_class[r['score']] for r in records], len(NATIVE_SCORES))
        arrays = {f'{s}_{k}': v for s, part in parts.items() for k, v in part.items()}
        np.savez_compressed(dest / f'{run.name}.npz', log_prior=prior, **arrays)
        print(f'{run.name}: val {len(parts["val"]["gt"])}, test {len(parts["test"]["gt"])}, '
              f'external {len(parts["ext"]["gt"])}; reproduces stored predictions ({ties} near-ties differ)', flush=True)


def to_levels(gt):
    return np.vectorize(LEVEL4.get)(gt)


def subset(z, split, keep=None):
    """{logits, y, pred, patient} of one split; keep = boolean mask."""
    out = {'logits': z[f'{split}_logits'], 'y': to_levels(z[f'{split}_gt']), 'patient': z[f'{split}_patient']}
    out['pred'] = to_levels(native_pred(out['logits']))
    return {k: v[keep] for k, v in out.items()} if keep is not None else out


def evaluate(cal, test, offset, n_boot, weights):
    """Point metrics and bootstrap replicates of one run in one scope."""
    temp = fit_temperature(cal['logits'], cal['y'], offset)
    p_cal = level_probs(cal['logits'], temp, offset)
    scores = expansion_mass(p_cal)[np.arange(len(cal['y'])), cal['y']]
    lam = conformal_lambda(scores)
    p_test = level_probs(test['logits'], temp, offset)
    prev_test = expansion_mass(p_test)
    point = tally(prev_test <= lam, test['y'], test['pred'])
    wc, cal_idx = weights(cal['patient'])
    wt, test_idx = weights(test['patient'])
    boot = np.empty((n_boot, len(METRICS)))
    for b in range(n_boot):
        lam_b = conformal_lambda(scores, weights=wc[b][cal_idx])
        boot[b] = tally(prev_test <= lam_b, test['y'], test['pred'], wt[b][test_idx])
    p_before = level_probs(test['logits'], 1.0, offset)
    calib = {'T': temp, 'lambda': lam, 'nll_before': nll(p_before, test['y']), 'nll_after': nll(p_test, test['y'])}
    for level in LEVELS:
        calib[f'ece_before_l{level}'] = ece(p_before[:, level], test['y'] == level)
        calib[f'ece_after_l{level}'] = ece(p_test[:, level], test['y'] == level)
    return point, boot, calib


class SharedWeights:
    """Multinomial patient weights, one set per patient list, so runs on the same patients stay paired."""

    def __init__(self, n_boot, seed):
        self.n_boot, self.rng, self.memo = n_boot, np.random.default_rng(seed), {}

    def __call__(self, patient):
        names, idx = np.unique(patient, return_inverse=True)
        key = tuple(names)
        if key not in self.memo:
            self.memo[key] = self.rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=self.n_boot)
        return self.memo[key], idx


def summarise(points, boots, calibs, rng, alpha=0.05, levels=None):
    """levels: (draw, seed, calibration draw) per entry; without it the entries are resampled on one level."""
    points, boots = np.stack(points), np.stack(boots, axis=1)       # R x M, B x R x M
    design = levels if levels is not None else flat_levels(boots.shape[1])
    pick = resample_indices(design, boots.shape[0], rng)
    hier = np.nanmean(np.take_along_axis(boots, pick[:, :, None], 1), axis=1)
    ci = np.nanpercentile(hier, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    calib = {k: float(np.mean([c[k] for c in calibs])) for k in calibs[0]}
    return {'runs': len(points), 'calibration': calib,
            **{m: {'mean': float(np.nanmean(points[:, i])), 'sd': float(np.nanstd(points[:, i])),
                   'ci': [float(ci[0, i]), float(ci[1, i])]} for m, i in IDX.items()}}


def scope_cells(run_name, z, draws, manifest_test):
    """{(scope, k): [(calibration, test) per calibration draw]} for one run."""
    shared = np.isin(z['ext_joint'], SHARED_JOINTS)
    in_test = np.isin(z['ext_patient'], manifest_test)
    ext_test = subset(z, 'ext', shared & in_test)
    cells = {('internal', 0): [(subset(z, 'val'), subset(z, 'test'))],
             ('external', 0): [(subset(z, 'val'), ext_test)]}
    for k in SIZES:
        cells[('local', k)] = [(subset(z, 'ext', shared & np.isin(z['ext_patient'], draws[d][k])), ext_test)
                               for d in DRAWS]
    return cells


def analyse(logit_dir, manifest, n_boot=2000, seed=0):
    m = pd.read_csv(manifest)
    manifest_test = m.loc[m['e4_split'] == 'test', 'patient'].unique()
    draws = {d: fewshot_patients(manifest, d) for d in DRAWS}
    rng = np.random.default_rng(seed)
    results = {}
    for c in CONFIGS:
        store, design, weights = {}, {}, SharedWeights(n_boot, seed)
        for run in source_runs(c):
            z = dict(np.load(Path(logit_dir) / f'{run.name}.npz'))
            run_tag = run_levels([run.name])[0]
            for (scope, k), pairs in scope_cells(run.name, z, draws, manifest_test).items():
                variants = ('raw', 'prior') if scope != 'local' else ('raw',)
                for variant in variants:
                    offset = z['log_prior'] if variant == 'prior' else None
                    for i, (cal, test) in enumerate(pairs):
                        res = evaluate(cal, test, offset, n_boot, weights)
                        slot = store.setdefault((scope, k, variant), ([], [], []))
                        for lst, val in zip(slot, res):
                            lst.append(val)
                        design.setdefault((scope, k, variant), []).append((*run_tag, i))
        for key, (points, boots, calibs) in store.items():
            results[(c, *key)] = summarise(points, boots, calibs, rng, levels=design[key])
        print(f'{c}: done', flush=True)
    return results


def h4(results):
    """The pre-set H4 criteria for each configuration (internal, primary variant)."""
    verdict = {}
    for c in CONFIGS:
        r = results[(c, 'internal', 0, 'raw')]
        verdict[c] = {'coverage': bool(r['coverage']['ci'][1] >= 1 - ALPHA),
                      'concentration': bool(r['enrichment']['ci'][0] > 1)}
    return verdict


def fmt(r, key, digits=3):
    return f"{r[key]['mean']:.{digits}f} [{r[key]['ci'][0]:.{digits}f}, {r[key]['ci'][1]:.{digits}f}]"


def table(results, scope, k, variant):
    lines = ['| Config | Runs | T | lambda | Coverage | Set size | Referred | Grade >= 2 referred | Enrichment | '
             'Grade >= 2 errors referred | Grade >= 2 in non-referred |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for c in CONFIGS:
        r = results[(c, scope, k, variant)]
        lines.append(f"| {c} {LABELS[c]} | {r['runs']} | {r['calibration']['T']:.2f} | {r['calibration']['lambda']:.3f} "
                     f"| {fmt(r, 'coverage')} | {fmt(r, 'set_size', 2)} | {fmt(r, 'referral')} | {fmt(r, 'sens_ge2')} | "
                     f"{fmt(r, 'enrichment', 2)} | {fmt(r, 'high_error_flagged')} | {fmt(r, 'prev_not_referred')} |")
    return lines


def report(results):
    lines = ['# E5: calibration, ordinal conformal sets and referral', '',
             'Target coverage 90%. 4 levels (0, 1, 2, >= 3). Mean over runs, 95% CI from a bootstrap that resamples '
             'calibration and test patients and redraws runs; lambda is recomputed per replicate.', '',
             '## H4 criteria (internal, raw logits)', '', '| Config | (a) Coverage CI reaches 0.90 | (b) Enrichment CI > 1 |',
             '|---|---|---|']
    for c, v in h4(results).items():
        lines.append(f"| {c} {LABELS[c]} | {'yes' if v['coverage'] else 'no'} | {'yes' if v['concentration'] else 'no'} |")
    titles = (('internal', 0, 'raw', 'Internal: RAM-H1200 val -> test, raw logits (primary)'),
              ('internal', 0, 'prior', 'Internal, prior restored (secondary)'),
              ('external', 0, 'raw', 'External: RAM-H1200 val -> RheumaVIT test, raw logits'),
              ('external', 0, 'prior', 'External, prior restored'),
              ('local', 5, 'raw', 'Local recalibration on 5 RheumaVIT patients (exploratory)'),
              ('local', 20, 'raw', 'Local recalibration on 20 RheumaVIT patients (exploratory)'))
    for scope, k, variant, title in titles:
        lines += ['', f'## {title}', '', *table(results, scope, k, variant)]
    lines += ['', '## Coverage per level', '', '| Config | Scope | Level 0 | Level 1 | Level 2 | Level >= 3 |',
              '|---|---|---|---|---|---|']
    for c in CONFIGS:
        for scope, k, variant, name in (('internal', 0, 'raw', 'internal'), ('external', 0, 'raw', 'external'),
                                        ('local', 20, 'raw', 'local 20')):
            r = results[(c, scope, k, variant)]
            lines.append(f"| {c} | {name} | " + ' | '.join(fmt(r, f'coverage_l{i}') for i in LEVELS) + ' |')
    lines += ['', '## Calibration on test (ECE per level, NLL)', '',
              '| Config | Scope | Variant | NLL before -> after | ECE level 0 | ECE level 1 | ECE level 2 | ECE level >= 3 |',
              '|---|---|---|---|---|---|---|---|']
    for c in CONFIGS:
        for scope, k, variant in (('internal', 0, 'raw'), ('internal', 0, 'prior'), ('external', 0, 'raw')):
            q = results[(c, scope, k, variant)]['calibration']
            eces = ' | '.join(f"{q[f'ece_before_l{i}']:.3f} -> {q[f'ece_after_l{i}']:.3f}" for i in LEVELS)
            lines.append(f"| {c} | {scope} | {variant} | {q['nll_before']:.3f} -> {q['nll_after']:.3f} | {eces} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('logits', 'analyse'))
    p.add_argument('--manifest', default='runs/e4/rheumavit/manifest.csv')
    p.add_argument('--e4', default='runs/e4')
    p.add_argument('--out', default='runs/e5')
    p.add_argument('--n-boot', type=int, default=2000)
    args = p.parse_args(argv)
    if args.step == 'logits':
        extract(args.manifest, args.e4, args.out)
        return
    results = analyse(Path(args.out) / 'logits', args.manifest, args.n_boot)
    text = report(results)
    dest = Path(args.out) / 'analysis'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'report.md').write_text(text)
    (dest / 'e5.json').write_text(json.dumps({'h4': h4(results), 'cells': {
        f'{c}_{s}_k{k}_{v}': r for (c, s, k, v), r in results.items()}}, indent=2))
    print(text)


if __name__ == '__main__':
    main()
