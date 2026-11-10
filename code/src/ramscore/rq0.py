"""RQ0: shared reference (RQ0-01, ResNet34 + ImageNet + balanced softmax) and the architecture check (RQ0-02,
ConvNeXt-Tiny fb_in1k on the same recipe), at 100% and 10% labels.

Post hoc: the RAM-H1200 test set was already used in the old repository. Contrasts, the Holm family and the verdict
rules were fixed before the first run in rules/rq0_prespecification.md. Paired patient bootstrap of compare.py with
runs redrawn (helpers of e6abl.py, as in e8.py).

Primary (Holm over 2): BACC4 RQ0-02 - RQ0-01 at 100% and at 10% labels.
Supporting (no verdict): RQ0-01 against the E8 `base` run of the same draw and seed in the old repository, plus the
share of joints with identical predictions and the per-run test BACC4 difference (|diff| >= 0.02 blocks RQ1).

Usage: uv run --group analysis python -m ramscore.rq0 [--runs ../runs] [--e8 <old runs/e8>] [--out ../runs/rq0_analysis]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import NATIVE_SCORES
from .e6abl import add_holm, contrast, load, paired
from .e7 import prediction_stats

ARMS = {'rq0-01': 'resnet34', 'rq0-02': 'convnext_tiny'}
LABELS = {'rq0-01': 'ResNet34 (rujukan)', 'rq0-02': 'ConvNeXt-Tiny'}
FRACTIONS = ('1.0', '0.1')
E8_DEFAULT = '/data/Smart-X/SLR-RA/experiments/runs/e8'
MIN_DIFF = 0.05
ALPHA = 0.05
REPRO_LIMIT = 0.02


def arm_glob(root, arm, frac):
    return str(Path(root) / 'rq0' / f'{arm}_{ARMS[arm]}_b2_f{frac}_d*_s*')


def e8_glob(e8_root, frac):
    return str(Path(e8_root) / f'e8_base_a1_resnet34_b2_f{frac}_d*_s*')


def verdict(stats):
    """Locked rules: a difference needs Holm p < .05, and it is meaningful from an absolute 0.05."""
    if stats['p_holm'] >= ALPHA:
        return 'tidak terbukti'
    return 'bermakna' if abs(stats['diff']) >= MIN_DIFF else 'kecil'


def run_summary(pattern):
    """Mean validation BACC4 and QWK and the selected epoch over the runs that have metrics.json."""
    paths = sorted(Path(p).parent / 'metrics.json' for p in load(pattern).values())
    ms = [json.loads(p.read_text()) for p in paths if p.exists()]
    if not ms:
        return None
    return {'runs': len(ms), 'val_bacc4': float(np.mean([m['val']['bacc4'] for m in ms])),
            'val_qwk': float(np.mean([m['val']['qwk'] for m in ms])),
            'best_epoch': float(np.mean([m['best_epoch'] for m in ms]))}


def reproduction(new_pattern, old_pattern):
    """Per (draw, seed): share of joints with the same prediction and the test BACC4 difference, new - old."""
    from .metrics import compute_metrics

    new, old = load(new_pattern), load(old_pattern)
    rows = []
    for tag in sorted(set(new) & set(old)):
        a, b = pd.read_csv(new[tag]), pd.read_csv(old[tag])
        merged = a.merge(b, on=['case', 'joint'], suffixes=('_new', '_old'))
        if len(merged) != len(a) or len(merged) != len(b) or not (merged['gt_new'] == merged['gt_old']).all():
            raise ValueError(f'{tag}: test joints or labels differ between the two runs')
        d = compute_metrics(a['gt'], a['pred'])['bacc4'] - compute_metrics(b['gt'], b['pred'])['bacc4']
        rows.append({'tag': tag, 'identical': float((merged['pred_new'] == merged['pred_old']).mean()),
                     'diff_bacc4': float(d), 'flag': bool(abs(d) >= REPRO_LIMIT)})
    return rows


def analyse(root, e8_root=E8_DEFAULT, n_boot=10000):
    root = Path(root)
    primary, supporting, repro, predictions, summaries = [], [], [], [], []
    for frac in FRACTIONS:
        pair = paired(arm_glob(root, 'rq0-02', frac), arm_glob(root, 'rq0-01', frac))
        if pair:
            primary.append({'frac': frac, **contrast(*pair, n_boot, icc=True)})
        if e8_root and Path(e8_root).exists():
            pair = paired(arm_glob(root, 'rq0-01', frac), e8_glob(e8_root, frac))
            if pair:
                supporting.append({'frac': frac, **contrast(*pair, n_boot, icc=True)})
                repro += [{'frac': frac, **r} for r in reproduction(arm_glob(root, 'rq0-01', frac), e8_glob(e8_root, frac))]
        for arm in ARMS:
            frames = [pd.read_csv(p) for p in load(arm_glob(root, arm, frac)).values()]
            if frames:
                predictions.append({'frac': frac, 'arm': arm, **prediction_stats(frames)})
            s = run_summary(arm_glob(root, arm, frac))
            if s:
                summaries.append({'frac': frac, 'arm': arm, **s})
    add_holm(primary)
    for r in primary:
        r['verdict'] = verdict(r['bacc4'])
    return {'primary': primary, 'supporting': supporting, 'reproduction': repro, 'predictions': predictions,
            'summaries': summaries}


def report(results):
    lines = ['# RQ0: rujukan bersama dan cek arsitektur (ResNet34 vs ConvNeXt-Tiny)', '',
             '*Post hoc*: test RAM-H1200 sudah dipakai di repo lama, sehingga hasil ini tidak dipakai untuk memilih '
             'konfigurasi tesis. Aturan dikunci sebelum run pertama: `rules/rq0_prespecification.md`. '
             'Holm dihitung atas 2 kontras primer BACC4; p-value lain tidak dikoreksi.', '',
             '## Kontras primer: ConvNeXt-Tiny − ResNet34 (rujukan)', '',
             '| Label | Run | BACC4 ResNet34 | BACC4 ConvNeXt-Tiny | Selisih [95% CI] | Holm p | Putusan | Selisih QWK | p | Selisih ICC | p |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results['primary']:
        a, q, i = r['bacc4'], r['qwk'], r['icc']
        lines.append(f"| {float(r['frac']):.0%} | {r['runs']} | {a['base']:.3f} | {a['variant']:.3f} | "
                     f"{a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {a['p_holm']:.3f} | {r['verdict']} | "
                     f"{q['diff']:+.3f} | {q['p']:.3f} | {i['diff']:+.3f} | {i['p']:.3f} |")
    lines += ['', '## Uji reproduksi: rujukan di repo ini vs run E8 `base` dengan draw/seed yang sama (tanpa putusan)', '',
              '| Label | Run | BACC4 lama | BACC4 baru | Selisih [95% CI] | p | Selisih QWK | p |', '|---|---|---|---|---|---|---|---|']
    for r in results['supporting']:
        a, q = r['bacc4'], r['qwk']
        lines.append(f"| {float(r['frac']):.0%} | {r['runs']} | {a['base']:.3f} | {a['variant']:.3f} | "
                     f"{a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {a['p']:.3f} | {q['diff']:+.3f} | {q['p']:.3f} |")
    if results['reproduction']:
        lines += ['', f'Per run (|selisih BACC4| ≥ {REPRO_LIMIT} = RQ1 ditahan sampai penyebabnya dicari):', '',
                  '| Label | Desain | Prediksi identik | Selisih BACC4 | Tanda |', '|---|---|---|---|---|']
        for r in results['reproduction']:
            lines.append(f"| {float(r['frac']):.0%} | {r['tag']} | {r['identical']:.3f} | {r['diff_bacc4']:+.3f} | "
                         f"{'**perlu dicek**' if r['flag'] else 'ok'} |")
    lines += ['', '## Validasi dan epoch terpilih (rata-rata run)', '',
              '| Label | Arm | Run | Val BACC4 | Val QWK | Epoch terpilih |', '|---|---|---|---|---|---|']
    for r in results['summaries']:
        lines.append(f"| {float(r['frac']):.0%} | {LABELS[r['arm']]} | {r['runs']} | {r['val_bacc4']:.3f} | "
                     f"{r['val_qwk']:.3f} | {r['best_epoch']:.1f} |")
    grades = [str(g) for g in NATIVE_SCORES]
    lines += ['', '## Recall per grade (test)', '', '| Label | Arm | Run | Proporsi prediksi grade 0 | ' +
              ' | '.join(f'Recall {g}' for g in grades) + ' |', '|---|---|---|---|' + '---|' * len(grades)]
    for r in results['predictions']:
        lines.append(f"| {float(r['frac']):.0%} | {LABELS[r['arm']]} | {r['runs']} | {r['pred_share_grade0']:.3f} | " +
                     ' | '.join(f"{r['recall'][g]:.3f}" for g in grades) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='../runs')
    p.add_argument('--e8', default=E8_DEFAULT, help='Old E8 run folder for the reproduction check ("" = skip)')
    p.add_argument('--out', default='../runs/rq0_analysis')
    p.add_argument('--n-boot', type=int, default=10000)
    args = p.parse_args(argv)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    results = analyse(args.runs, args.e8, args.n_boot)
    text = report(results)
    (dest / 'report.md').write_text(text)
    (dest / 'rq0.json').write_text(json.dumps(results, indent=2))
    print(text)


if __name__ == '__main__':
    main()
