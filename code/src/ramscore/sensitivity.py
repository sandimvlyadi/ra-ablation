"""Post hoc analyses on stored predictions and training logs; nothing is retrained.

Added after the first results to answer review questions, so every result here is exploratory.

  rq2       B2, B4 and B6 - B1 per encoder x fraction on BACC4, QWK and hand ICC(A,1) (Holm over 3, per metric)
  hand      signed hand-total error, limits of agreement and absolute error per E3 cell, whole test set and the
            hands with the lowest 75% of reference totals
  frozen    fine-tuned - frozen linear probe per encoder, head and fraction; the two arms differ in protocol,
            classifier, class balancing and selection metric, so this is a description and not a causal contrast
  recipe    ViT-S: each encoder under its recipe (uniform or LLRD) chosen on validation BACC4
  planned   ICC(3,1) of the patient total, the estimand first planned for the hand-level endpoint
  b1        training logs and predicted grades of B1 per encoder
  selection validation BACC4 at the validation-QWK checkpoint against the best validation BACC4 epoch; only the
            QWK checkpoint was stored, so test metrics cannot be re-selected
  budget    scratch ResNet34 at 100%: 200 against 50 epochs; where the best epoch falls in the training budget
  centres   composition and metrics per centre
  nonra     non-RA patients per split and per label draw

A replicate resamples test patients once and redraws runs along the training design (compare.resample_indices).
Arms on one design redraw the same runs. Frozen and fine-tuned arms have different designs and are redrawn apart.

Usage: uv run --group analysis python -m ramscore.sensitivity --out runs/sensitivity_analysis
"""
import argparse
import glob
import json
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import e3
from .compare import RunPicks, holm, metrics_from_confusion, patient_confusions, run_levels, two_sided_p
from .data import SCORE_FILE, parse_case, subsample_patients
from .e2 import ENCODERS as E2_ENCODERS
from .icc import ci, hand_totals, icc_a1

METRICS = ('bacc4', 'qwk', 'icc')
FROZEN = {
    ('resnet34', 'a1'): 'imagenet_resnet34', ('resnet34', 'a4'): 'e1_dino_resnet34_ep100',
    ('resnet34', 'a5'): 'e1_dinochrono_resnet34_ep100', ('vits', 'a1'): 'imagenet_vit_small_patch16_224',
    ('vits', 'a3'): 'e1_mae_vits_ep400', ('vits', 'a4'): 'e1_dino_vits_ep100', ('vits', 'a5'): 'e1_dinochrono_vits_ep100',
}
XLSX = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
CONFIG_CENTRES = (('a0', 'b2', '1.0'), ('a1', 'b2', '1.0'), ('a4', 'b2', '1.0'), ('a1', 'b4', '1.0'), ('a4', 'b2', '0.1'))
TAG = re.compile(r'(d\d_s\d+)$')


def icc_c1_weighted(x1, x2, w=None):
    """ICC(C,1) = ICC(3,1) with frequency weights of shape (n,) or (B, n) -> (B,)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    w = np.ones((1, len(x1))) if w is None else np.atleast_2d(w).astype(float)
    n = w.sum(1)
    r, e = (x1 + x2) / 2, (x1 - x2) / 2
    mean_r, mean_e = w @ r / n, w @ e / n
    msr = 2 * (w @ r ** 2 - n * mean_r ** 2) / (n - 1)
    mse = 2 * (w @ e ** 2 - n * mean_e ** 2) / (n - 1)
    return (msr - mse) / (msr + mse)


def load_arm(pattern):
    """(prediction frames, (draw, seed) design) of the runs matching a run-folder pattern."""
    paths = sorted(glob.glob(f'{pattern}/test_predictions.csv'))
    if not paths:
        raise FileNotFoundError(pattern)
    return [pd.read_csv(p) for p in paths], run_levels([Path(p).parent.name for p in paths])


class Scorer:
    """Metrics of prediction arms with shared bootstrap replicates of one test set."""

    def __init__(self, reference, n_boot, seed=0):
        self.patients = sorted(reference['patient'].unique())
        self.n_boot = n_boot
        self.picks = RunPicks(n_boot, seed)
        rng = np.random.default_rng(seed)
        self.w = rng.multinomial(len(self.patients), np.full(len(self.patients), 1 / len(self.patients)), size=n_boot)

    def case_w(self, totals):
        index = pd.Index(self.patients).get_indexer(totals['patient'])
        return self.w[:, index]

    def score(self, runs, levels, picks=None):
        """{metric: (point, draws)} on the mean over runs."""
        chosen = (picks or self.picks)(levels)
        cms = np.stack([patient_confusions(df, self.patients) for df in runs])
        full = metrics_from_confusion(cms.sum(1))
        boot = metrics_from_confusion(np.einsum('bp,rpij->brij', self.w, cms))
        out = {k: (float(full[k].mean()), np.take_along_axis(boot[k], chosen, 1).mean(1)) for k in ('bacc4', 'qwk')}
        totals = [hand_totals(df) for df in runs]
        w = self.case_w(totals[0])
        icc = np.stack([icc_a1(t['gt'], t['pred'], w) for t in totals], 1)
        out['icc'] = (float(np.mean([icc_a1(t['gt'], t['pred'])[0] for t in totals])),
                      np.take_along_axis(icc, chosen, 1).mean(1))
        return out


def contrast(a, b, metric):
    """Difference of two scored arms: diff, CI, two-sided p."""
    d = a[metric][1] - b[metric][1]
    d = d[~np.isnan(d)]
    return {'diff': a[metric][0] - b[metric][0], 'ci': ci(d), 'p': two_sided_p(d)}


def add_holm(rows, metric):
    for row, p in zip(rows, holm(np.array([r[metric]['p'] for r in rows]))):
        row[metric]['p_holm'] = float(p)


def rq2_family(root, scorer):
    rows = []
    for frac in e3.FRACTIONS:
        for a in e3.ENCODERS:
            arms = {b: scorer.score(*load_arm(f'{root}/{e3.run_dir(a, b)}_f{frac}_d*_s*')) for b in ('b1', 'b2', 'b4', 'b6')}
            family = [{'frac': frac, 'encoder': a, 'contrast': f'{b} - b1',
                       **{m: contrast(arms[b], arms['b1'], m) for m in METRICS}} for b in ('b2', 'b4', 'b6')]
            for m in METRICS:
                add_holm(family, m)
            rows += family
    return rows


def hand_error_stats(totals, w, picks):
    """Signed error, absolute error and limits of agreement over the hands in totals, with CIs on the first two."""
    error = np.stack([(t['pred'] - t['gt']).to_numpy(float) for t in totals])      # (runs, hands)
    n = w.sum(1, keepdims=True)
    bias = (w @ error.T) / n
    abs_error = (w @ np.abs(error).T) / n
    point_bias, point_sd = error.mean(1), error.std(1, ddof=1)
    return {
        'bias': float(point_bias.mean()), 'bias_ci': ci(np.take_along_axis(bias, picks, 1).mean(1)),
        'abs_error': float(np.abs(error).mean()), 'abs_error_ci': ci(np.take_along_axis(abs_error, picks, 1).mean(1)),
        'loa': [float((point_bias - 1.96 * point_sd).mean()), float((point_bias + 1.96 * point_sd).mean())],
    }


def hand_analysis(root, scorer, share=0.75):
    """Hand-total error per E3 cell, on all hands and on the hands up to the given quantile of the reference total."""
    rows = []
    for frac in e3.FRACTIONS:
        for a in e3.ENCODERS:
            for b in ('b0', 'b1', 'b2', 'b4', 'b6'):
                runs, levels = load_arm(f'{root}/{e3.run_dir(a, b)}_f{frac}_d*_s*')
                totals = [hand_totals(df) for df in runs]
                gt = totals[0]['gt'].to_numpy(float)
                cut = np.quantile(gt, share)
                low = gt <= cut
                w, picks = scorer.case_w(totals[0]), scorer.picks(levels)
                boot = np.stack([icc_a1(t['gt'].to_numpy()[low], t['pred'].to_numpy()[low], w[:, low]) for t in totals], 1)
                rows.append({
                    'frac': frac, 'encoder': a, 'head': b, 'all': hand_error_stats(totals, w, picks),
                    'low': {
                        **hand_error_stats([t[low] for t in totals], w[:, low], picks),
                        'icc': float(np.mean([icc_a1(t['gt'].to_numpy()[low], t['pred'].to_numpy()[low])[0] for t in totals])),
                        'icc_ci': ci(np.take_along_axis(boot, picks, 1).mean(1)),
                    },
                    'cut': float(cut), 'n_low': int(low.sum()), 'n_all': len(gt),
                    'sd_all': float(gt.std(ddof=1)), 'sd_low': float(gt[low].std(ddof=1)),
                    'naive_abs_error_low': float(np.abs(gt[low] - np.median(gt[low])).mean()),
                })
    return rows


def frozen_vs_fine(root, scorer):
    """Fine-tuned minus frozen linear probe. The two arms redraw runs apart because their designs differ."""
    other = RunPicks(scorer.n_boot, 1)
    rows = []
    for (backbone, a), name in FROZEN.items():
        for frac in ('0.1', '1.0'):
            frozen = scorer.score(*load_arm(f'{root}/e1_probe/pred/{name}/linear_f{frac}*'), picks=other)
            for b in ('b0', 'b4', 'b6'):
                fine = scorer.score(*load_arm(f'{root}/e2/e2_{a}_{backbone}_{b}_f{frac}_d*_s*'))
                rows.append({'backbone': backbone, 'encoder': a, 'frac': frac, 'head': b,
                             **{f'frozen_{m}': frozen[m][0] for m in METRICS}, **{f'fine_{m}': fine[m][0] for m in METRICS},
                             **{m: contrast(fine, frozen, m) for m in METRICS}})
    return rows


def val_mean(pattern, tags):
    """Mean validation BACC4 of the runs whose draw/seed tag is in tags."""
    values = [json.loads((Path(p) / 'metrics.json').read_text())['val']['bacc4']
              for p in sorted(glob.glob(pattern)) if TAG.search(p)[1] in tags]
    return float(np.mean(values))


def recipe_selected(root, scorer):
    """Each ViT-S encoder under the recipe with the higher mean validation BACC4 of that fraction x head cell."""
    encoders = {'mae': 'a3', 'imagenet': 'a1', 'dino': 'a4'}
    rows, chosen = [], []
    for frac in ('0.1', '1.0'):
        for b in ('b0', 'b4', 'b6'):
            arms = {}
            for label, a in encoders.items():
                llrd = f'{root}/e6/e6_{a}_vits_{b}_f{frac}_d*_s*'
                uniform = f'{root}/e2/e2_{a}_vits_{b}_f{frac}_d*_s*'
                tags = {TAG.search(p)[1] for p in glob.glob(llrd)}
                val_llrd, val_uniform = val_mean(llrd, tags), val_mean(uniform, tags)
                recipe = 'llrd' if val_llrd >= val_uniform else 'e2'
                chosen.append({'frac': frac, 'head': b, 'encoder': label, 'val_llrd': val_llrd, 'val_e2': val_uniform, 'recipe': recipe})
                arms[label] = scorer.score(*load_arm(llrd if recipe == 'llrd' else uniform))
            for label in ('mae', 'dino'):
                rows.append({'frac': frac, 'head': b, 'contrast': f'{label} - imagenet',
                             **{f'{label}_{m}': arms[label][m][0] for m in METRICS},
                             **{f'imagenet_{m}': arms['imagenet'][m][0] for m in METRICS},
                             **{m: contrast(arms[label], arms['imagenet'], m) for m in METRICS}})
    for m in METRICS:
        add_holm(rows, m)
    return chosen, rows


def planned_estimand(root, scorer):
    """ICC(3,1) of the patient total (sum over all test hands of a patient) on the planned contrasts."""
    def cell(pattern):
        runs, levels = load_arm(pattern)
        totals = [hand_totals(df).groupby('patient').agg(gt=('gt', 'sum'), pred=('pred', 'sum')) for df in runs]
        w = scorer.w[:, pd.Index(scorer.patients).get_indexer(totals[0].index)]
        boot = np.stack([icc_c1_weighted(t['gt'], t['pred'], w) for t in totals], 1)
        point = float(np.mean([icc_c1_weighted(t['gt'], t['pred'])[0] for t in totals]))
        return point, np.take_along_axis(boot, scorer.picks(levels), 1).mean(1)

    def pair(label, a, b):
        d = a[1] - b[1]
        return {'contrast': label, 'a': a[0], 'b': b[0], 'diff': a[0] - b[0], 'ci': ci(d), 'p': two_sided_p(d[~np.isnan(d)])}

    e2_rows = []
    for backbone, ssl in (('resnet34', ('a4',)), ('vits', ('a3', 'a4'))):
        for s in ssl:
            for r in ('a0', 'a1'):
                for b in ('b0', 'b4'):
                    e2_rows.append(pair(f'{backbone} {b}: {s} - {r}', cell(f'{root}/e2/e2_{s}_{backbone}_{b}_f0.1_d*_s*'),
                                        cell(f'{root}/e2/e2_{r}_{backbone}_{b}_f0.1_d*_s*')))
    e3_rows = [pair(f'{a}: b4 - b1', cell(f"{root}/{e3.run_dir(a, 'b4')}_f1.0_d*_s*"),
                    cell(f"{root}/{e3.run_dir(a, 'b1')}_f1.0_d*_s*")) for a in e3.ENCODERS]
    for family in (e2_rows, e3_rows):
        for row, p in zip(family, holm(np.array([r['p'] for r in family]))):
            row['p_holm'] = float(p)
    return {'e2': e2_rows, 'e3': e3_rows}


def run_log(run):
    """Summary of one run from metrics.json and history.csv."""
    run = Path(run)
    metrics = json.loads((run / 'metrics.json').read_text())
    history = pd.read_csv(run / 'history.csv')
    best = history[history['epoch'] == metrics['best_epoch']].iloc[0]
    return {
        'epochs': len(history), 'best_epoch': int(metrics['best_epoch']), 'val_qwk': float(best['val_qwk']),
        'val_bacc4_at_qwk': float(best['val_bacc4']), 'val_bacc4_best': float(history['val_bacc4'].max()),
        'val_mae_at_best': float(best['val_mae']), 'val_mae_min': float(history['val_mae'].min()),
        'val_mae_last': float(history['val_mae'].iloc[-1]), 'val_qwk_max': float(history['val_qwk'].max()),
        'share_epochs_mae_above_1': float((history['val_mae'] > 1).mean()),
        'val_recall_ge2': float(metrics['val']['recall_ge2']), 'test_recall_ge2': float(metrics['test']['recall_ge2']),
    }


def b1_logs(root):
    rows = []
    for frac in e3.FRACTIONS:
        for a in e3.ENCODERS:
            logs, shares = [], []
            for run in sorted(glob.glob(f"{root}/{e3.run_dir(a, 'b1')}_f{frac}_d*_s*")):
                logs.append(run_log(run))
                pred = pd.read_csv(Path(run) / 'test_predictions.csv')['pred']
                shares.append({g: float((pred == g).mean()) for g in (0, 1, 2, 3, 5)})
            rows.append({'frac': frac, 'encoder': a, 'runs': len(logs),
                         **{k: float(np.mean([log[k] for log in logs])) for k in logs[0]},
                         'pred_share': {g: float(np.mean([s[g] for s in shares])) for g in (0, 1, 2, 3, 5)}})
    return rows


def selection_logs(root):
    """Validation BACC4 at the stored checkpoint against the best validation BACC4 epoch, on the E2 primary cells."""
    rows = []
    for backbone, encoders in E2_ENCODERS.items():
        for a in encoders:
            for b in ('b0', 'b4'):
                for frac in ('0.1', '1.0'):
                    logs = [run_log(r) for r in sorted(glob.glob(f'{root}/e2/e2_{a}_{backbone}_{b}_f{frac}_d*_s*'))]
                    if logs:
                        rows.append({'backbone': backbone, 'encoder': a, 'head': b, 'frac': frac, 'runs': len(logs),
                                     'at_qwk': float(np.mean([log['val_bacc4_at_qwk'] for log in logs])),
                                     'best': float(np.mean([log['val_bacc4_best'] for log in logs]))})
    return rows


def mean_test(paths):
    metrics = [json.loads(Path(p).read_text())['test'] for p in paths]
    return {'bacc4': float(np.mean([m['bacc4'] for m in metrics])), 'qwk': float(np.mean([m['qwk'] for m in metrics])), 'runs': len(metrics)}


def budget_logs(root):
    """Scratch ResNet34 at 100% (200 and 50 epochs) and the position of the best epoch per encoder x fraction."""
    long_run = [{'head': b, **mean_test(sorted(glob.glob(f'{root}/{b}_resnet34_f1.0_s*/metrics.json')))} for b in e3.STRATEGIES]
    short_run = [{'head': b, **mean_test(sorted(glob.glob(f"{root}/{e3.run_dir('a0', b)}_f1.0_d*_s*/metrics.json")))} for b in e3.STRATEGIES]
    position = []
    for frac in e3.FRACTIONS:
        for a in e3.ENCODERS:
            for b in e3.STRATEGIES:
                logs = [run_log(r) for r in sorted(glob.glob(f'{root}/{e3.run_dir(a, b)}_f{frac}_d*_s*'))]
                position.append({'frac': frac, 'encoder': a, 'head': b, 'epochs': float(np.mean([log['epochs'] for log in logs])),
                                 'best_epoch': float(np.mean([log['best_epoch'] for log in logs])),
                                 'best_in_last_fifth': float(np.mean([log['best_epoch'] >= 0.8 * log['epochs'] for log in logs]))})
    return {'epochs200': long_run, 'epochs50': short_run, 'position': position}


def split_records(data_root, split):
    """Joint records of one split from its score file: patient, centre, case and score."""
    scores = json.loads((Path(data_root) / split / SCORE_FILE).read_text())
    records = []
    for name, joints in scores.items():
        case = parse_case(Path(name).stem)
        records += [{'patient': case.patient, 'centre': case.centre, 'case': Path(name).stem, 'score': int(s)}
                    for s in joints.values()]
    return records


def centre_composition(data_root):
    rows = []
    for split in ('train', 'val', 'test'):
        df = pd.DataFrame(split_records(data_root, split))
        for centre, group in df.groupby('centre'):
            rows.append({'split': split, 'centre': centre, 'patients': group['patient'].nunique(),
                         'hands': group['case'].nunique(), 'joints': len(group),
                         'share_hands': group['case'].nunique() / df['case'].nunique(),
                         'ge2': float((group['score'] >= 2).mean())})
    return rows


def centre_metrics(root, n_boot):
    rows = []
    for a, b, frac in CONFIG_CENTRES:
        runs, levels = load_arm(f'{root}/{e3.run_dir(a, b)}_f{frac}_d*_s*')
        for df in runs:
            df['centre'] = df['patient'].str.split('_').str[0]
        for centre in sorted(runs[0]['centre'].unique()):
            subset = [df[df['centre'] == centre].reset_index(drop=True) for df in runs]
            scorer = Scorer(subset[0], n_boot)
            scored = scorer.score(subset, levels)
            rows.append({'config': f'{a}_{b}_{frac}', 'centre': centre, 'patients': len(scorer.patients),
                         'hands': subset[0]['case'].nunique(), 'joints': len(subset[0]),
                         **{m: scored[m][0] for m in METRICS}, **{f'{m}_ci': ci(scored[m][1]) for m in METRICS}})
    return rows


def read_metadata(path):
    """Rows of the first sheet of an .xlsx file as a frame, without an extra dependency."""
    with zipfile.ZipFile(path) as z:
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(t.text or '' for t in si.iter(f'{XLSX}t')) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
    rows = []
    for row in sheet.iter(f'{XLSX}row'):
        cells = {}
        for c in row:
            column = re.match(r'[A-Z]+', c.get('r'))[0]
            value, inline = c.find(f'{XLSX}v'), c.find(f'{XLSX}is')
            if inline is not None:
                cells[column] = ''.join(t.text or '' for t in inline.iter(f'{XLSX}t'))
            elif value is not None:
                cells[column] = shared[int(value.text)] if c.get('t') == 's' else value.text
        rows.append(cells)
    frame = pd.DataFrame(rows)
    frame.columns = frame.iloc[0]
    return frame.iloc[1:].reset_index(drop=True)


def nonra(data_root, metadata):
    meta = read_metadata(metadata)
    stems = meta['Mapped Image Stem'].str.split('_')
    meta['patient'] = stems.str[1] + '_' + stems.str[2]
    flagged = meta[meta['isRA'].astype(float) == 0]
    ids = set(flagged['patient'])
    out = {'rows': len(meta), 'rows_nonra': len(flagged), 'unique_ids_nonra': len(ids),
           'centres_nonra': flagged.groupby(stems.str[1]).size().to_dict(), 'splits': [], 'draws': []}
    train = split_records(data_root, 'train')
    for split in ('train', 'val', 'test'):
        df = pd.DataFrame(split_records(data_root, split))
        df['nonra'] = df['patient'].isin(ids)
        out['splits'].append({'split': split, 'patients': df['patient'].nunique(),
                              'patients_nonra': df.loc[df['nonra'], 'patient'].nunique(), 'hands': df['case'].nunique(),
                              'hands_nonra': df.loc[df['nonra'], 'case'].nunique(), 'joints': len(df),
                              'joints_nonra': int(df['nonra'].sum()),
                              'grades_nonra': {int(k): int(v) for k, v in df.loc[df['nonra'], 'score'].value_counts().items()}})
    for fraction in (0.01, 0.05, 0.1, 0.25):
        for draw in range(3):
            kept = pd.DataFrame(subsample_patients(train, fraction, draw))
            kept['nonra'] = kept['patient'].isin(ids)
            out['draws'].append({'fraction': fraction, 'draw': draw, 'patients': kept['patient'].nunique(),
                                 'patients_nonra': kept.loc[kept['nonra'], 'patient'].nunique(), 'joints': len(kept),
                                 'joints_nonra': int(kept['nonra'].sum())})
    return out


def fmt(c):
    return f"{c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"


def pct(frac):
    return f'{float(frac):.0%}'


def table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]


def report(res):
    lines = ['# Post hoc analyses on stored outputs', '', 'Exploratory; no retraining. See the docstring of ramscore/sensitivity.py.']
    lines += ['', '## RQ2: B2, B4, B6 - B1 (Holm over 3 within encoder x fraction, per metric)', '']
    lines += table(['Labels', 'Encoder', 'Contrast', 'BACC4', 'Holm', 'QWK', 'Holm', 'Hand ICC(A,1)', 'Holm'],
                   [[pct(r['frac']), r['encoder'], r['contrast'], fmt(r['bacc4']), f"{r['bacc4']['p_holm']:.3f}", fmt(r['qwk']),
                     f"{r['qwk']['p_holm']:.3f}", fmt(r['icc']), f"{r['icc']['p_holm']:.3f}"] for r in res['rq2']])
    lines += ['', '## Hand-total error (score units; prediction - reference)', '']
    lines += table(['Labels', 'Encoder', 'Head', 'Bias [95% CI]', 'Limits of agreement', 'Abs. error [95% CI]',
                    'Low: hands', 'Low: ICC(A,1) [95% CI]', 'Low: bias [95% CI]', 'Low: abs. error [95% CI]', 'Low: naive abs. error'],
                   [[pct(r['frac']), r['encoder'], r['head'],
                     f"{r['all']['bias']:+.2f} [{r['all']['bias_ci'][0]:+.2f}, {r['all']['bias_ci'][1]:+.2f}]",
                     f"{r['all']['loa'][0]:+.1f}, {r['all']['loa'][1]:+.1f}",
                     f"{r['all']['abs_error']:.2f} [{r['all']['abs_error_ci'][0]:.2f}, {r['all']['abs_error_ci'][1]:.2f}]",
                     f"{r['n_low']}/{r['n_all']} (total <= {r['cut']:.0f})",
                     f"{r['low']['icc']:.3f} [{r['low']['icc_ci'][0]:.3f}, {r['low']['icc_ci'][1]:.3f}]",
                     f"{r['low']['bias']:+.2f} [{r['low']['bias_ci'][0]:+.2f}, {r['low']['bias_ci'][1]:+.2f}]",
                     f"{r['low']['abs_error']:.2f} [{r['low']['abs_error_ci'][0]:.2f}, {r['low']['abs_error_ci'][1]:.2f}]",
                     f"{r['naive_abs_error_low']:.2f}"] for r in res['hand']])
    first = res['hand'][0]
    lines += ['', f"Reference hand total SD: all hands {first['sd_all']:.2f}, low stratum {first['sd_low']:.2f}."]
    lines += ['', '## Fine-tuned - frozen linear probe', '']
    lines += table(['Backbone', 'Encoder', 'Labels', 'Head', 'BACC4 frozen', 'BACC4 fine', 'Diff', 'QWK frozen', 'QWK fine', 'Diff',
                    'ICC frozen', 'ICC fine', 'Diff'],
                   [[r['backbone'], r['encoder'], pct(r['frac']), r['head'], f"{r['frozen_bacc4']:.3f}", f"{r['fine_bacc4']:.3f}",
                     fmt(r['bacc4']), f"{r['frozen_qwk']:.3f}", f"{r['fine_qwk']:.3f}", fmt(r['qwk']), f"{r['frozen_icc']:.3f}",
                     f"{r['fine_icc']:.3f}", fmt(r['icc'])] for r in res['frozen']])
    lines += ['', '## ViT-S: recipe chosen per encoder on validation BACC4', '']
    lines += table(['Labels', 'Head', 'Encoder', 'Val uniform', 'Val LLRD', 'Chosen'],
                   [[pct(r['frac']), r['head'], r['encoder'], f"{r['val_e2']:.3f}", f"{r['val_llrd']:.3f}", r['recipe']]
                    for r in res['recipe_chosen']])
    lines += [''] + table(['Labels', 'Head', 'Contrast', 'BACC4', 'p', 'Holm', 'QWK', 'Holm', 'ICC', 'Holm'],
                          [[pct(r['frac']), r['head'], r['contrast'], fmt(r['bacc4']), f"{r['bacc4']['p']:.3f}", f"{r['bacc4']['p_holm']:.3f}",
                            fmt(r['qwk']), f"{r['qwk']['p_holm']:.3f}", fmt(r['icc']), f"{r['icc']['p_holm']:.3f}"] for r in res['recipe']])
    lines += ['', '## Planned estimand: ICC(3,1) of the patient total', '']
    for name, rows in res['planned'].items():
        lines += table([f'{name} contrast', 'First', 'Second', 'Diff [95% CI]', 'Holm'],
                       [[r['contrast'], f"{r['a']:.3f}", f"{r['b']:.3f}", fmt(r), f"{r['p_holm']:.3f}"] for r in rows]) + ['']
    lines += ['## B1 training logs and predicted grades', '']
    lines += table(['Labels', 'Encoder', 'Runs', 'Best epoch', 'Epochs', 'Val QWK', 'Val QWK max', 'Val MAE at best', 'Val MAE min', 'Val MAE last',
                    'Epochs with val MAE > 1', 'Test recall >=2', 'Predicted 0 / 1 / 2 / 3 / 5'],
                   [[pct(r['frac']), r['encoder'], r['runs'], f"{r['best_epoch']:.1f}", f"{r['epochs']:.0f}", f"{r['val_qwk']:.3f}",
                     f"{r['val_qwk_max']:.3f}", f"{r['val_mae_at_best']:.2f}", f"{r['val_mae_min']:.2f}", f"{r['val_mae_last']:.2f}",
                     f"{r['share_epochs_mae_above_1']:.0%}", f"{r['test_recall_ge2']:.3f}",
                     ' / '.join(f"{r['pred_share'][g]:.3f}" for g in (0, 1, 2, 3, 5))] for r in res['b1']])
    lines += ['', '## Checkpoint selection: validation BACC4 at the QWK checkpoint vs best epoch', '']
    lines += table(['Backbone', 'Encoder', 'Head', 'Labels', 'Runs', 'At QWK checkpoint', 'Best epoch', 'Gap'],
                   [[r['backbone'], r['encoder'], r['head'], pct(r['frac']), r['runs'], f"{r['at_qwk']:.3f}", f"{r['best']:.3f}",
                     f"{r['best'] - r['at_qwk']:.3f}"] for r in res['selection']])
    gaps = [r['best'] - r['at_qwk'] for r in res['selection']]
    lines += ['', f'Gap over {len(gaps)} cells: median {statistics.median(gaps):.3f}, largest {max(gaps):.3f}.']
    lines += ['', '## Budget: scratch ResNet34 at 100%, 200 against 50 epochs', '']
    lines += table(['Head', '200 epochs BACC4', '50 epochs BACC4', '200 epochs QWK', '50 epochs QWK'],
                   [[x['head'], f"{x['bacc4']:.3f}", f"{y['bacc4']:.3f}", f"{x['qwk']:.3f}", f"{y['qwk']:.3f}"]
                    for x, y in zip(res['budget']['epochs200'], res['budget']['epochs50'])])
    lines += ['', '## Best epoch position in the budget', '']
    lines += table(['Labels', 'Encoder', 'Head', 'Epochs', 'Best epoch', 'Runs with best epoch in the last fifth'],
                   [[pct(r['frac']), r['encoder'], r['head'], f"{r['epochs']:.0f}", f"{r['best_epoch']:.1f}", f"{r['best_in_last_fifth']:.0%}"]
                    for r in res['budget']['position']])
    lines += ['', '## Centres', '']
    lines += table(['Split', 'Centre', 'Patients', 'Hands', 'Joints', 'Share of hands', 'Grade >= 2'],
                   [[r['split'], r['centre'], r['patients'], r['hands'], r['joints'], f"{r['share_hands']:.1%}", f"{r['ge2']:.2%}"] for r in res['centres']])
    lines += [''] + table(['Config', 'Centre', 'Patients', 'Hands', 'BACC4 [95% CI]', 'QWK [95% CI]', 'Hand ICC(A,1) [95% CI]'],
                          [[r['config'], r['centre'], r['patients'], r['hands'],
                            *[f"{r[m]:.3f} [{r[f'{m}_ci'][0]:.3f}, {r[f'{m}_ci'][1]:.3f}]" for m in METRICS]] for r in res['centre_metrics']])
    n = res['nonra']
    lines += ['', '## Non-RA patients (Metadata.xlsx, isRA = 0)', '',
              f"{n['rows_nonra']} of {n['rows']} rows, {n['unique_ids_nonra']} centre-qualified patient identifiers; per centre {n['centres_nonra']}.", '']
    lines += table(['Split', 'Patients', 'Non-RA', 'Hands', 'Non-RA', 'Joints', 'Non-RA', 'Grades of non-RA joints'],
                   [[r['split'], r['patients'], r['patients_nonra'], r['hands'], r['hands_nonra'], r['joints'], r['joints_nonra'], r['grades_nonra']] for r in n['splits']])
    lines += [''] + table(['Fraction', 'Draw', 'Patients', 'Non-RA', 'Joints', 'Non-RA'],
                          [[f"{r['fraction']:.0%}", r['draw'], r['patients'], r['patients_nonra'], r['joints'], r['joints_nonra']] for r in n['draws']])
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='runs/sensitivity_analysis')
    p.add_argument('--data-root', default='/data/Smart-X/tesis-fix/dataset/RAM-H1200-v1/SvdH_Scoring/SvdH_BE_Scoring')
    p.add_argument('--metadata', default='/data/Smart-X/tesis-fix/dataset/RAM-H1200-v1/Metadata.xlsx')
    p.add_argument('--n-boot', type=int, default=10000)
    args = p.parse_args(argv)
    root = args.runs
    reference = load_arm(f"{root}/{e3.run_dir('a0', 'b0')}_f1.0_d*_s*")[0][0]
    scorer = Scorer(reference, args.n_boot)
    chosen, recipes = recipe_selected(root, scorer)
    res = {'rq2': rq2_family(root, scorer), 'hand': hand_analysis(root, scorer), 'frozen': frozen_vs_fine(root, scorer),
           'recipe_chosen': chosen, 'recipe': recipes, 'planned': planned_estimand(root, scorer), 'b1': b1_logs(root),
           'selection': selection_logs(root), 'budget': budget_logs(root), 'centres': centre_composition(args.data_root),
           'centre_metrics': centre_metrics(root, args.n_boot), 'nonra': nonra(args.data_root, args.metadata)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = report(res)
    (out / 'report.md').write_text(text)
    (out / 'sensitivity.json').write_text(json.dumps(res, indent=2, default=str))
    print(text)


if __name__ == '__main__':
    main()
