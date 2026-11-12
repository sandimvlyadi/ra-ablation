"""RQ1: pre-training and SSL against the ImageNet ResNet34 reference (RQ0-01), at 100% and 10% labels.

Post hoc: the RAM-H1200 test set was already used in the old repository. Arms, contrasts, the Holm families, the
validation-based selections and the verdict rules were fixed before the first run in rules/rq1_prespecification.md.
Paired patient bootstrap of compare.py with runs redrawn (helpers of e6abl.py, as in rq0.py).

Families (one Holm family each, fixed size; both label fractions in the same family; a contrast whose arm has not
run is reported as "tidak dijalankan" and enters Holm with p = 1):
  F1  pre-training / SSL - RQ0-01                       8 x 2 = 16
  F2  between methods and architectures                 7 x 2 = 14
  F3  modifications (vs original and vs best existing)  4 x 2 = 8
  F4  fine-tuning strategy                              5 x 2 = 10
  F5  priority-B arms: size fixed when they are decided, so it is not computed here.

Selections use the mean validation BACC4 of the arm's runs, never test. "Best existing" is chosen per label fraction
among the finished candidates; "best SSL" over the 6 runs of each candidate. A selection is provisional while any
candidate has not finished, and contrasts that depend on a provisional selection are not computed.

Secondary (no correction): QWK and hand-total ICC differences, recall per grade, share of grade-0 predictions,
validation BACC4/QWK, selected epoch, mean test - validation BACC4 (descriptive check of test optimism), and the
train - validation gap of the selected checkpoint (`gap`, GPU, as e8.py gap).

Usage: uv run --group analysis python -m ramscore.rq1 gap   # train/val scores of every checkpoint; GPU
       uv run --group analysis python -m ramscore.rq1       # contrasts and report; reads gap.json when it exists
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import NATIVE_SCORES
from .e6abl import add_holm, contrast, load, paired
from .e7 import prediction_stats

FRACTIONS = ('1.0', '0.1')
MIN_DIFF = 0.05
ALPHA = 0.05
BEST = 'best_existing'
BEST_NO_SKELEX = 'best_existing_no_skelex'
BEST_SSL = 'best_ssl'
SELECTIONS = (BEST, BEST_NO_SKELEX, BEST_SSL)

LABELS = {
    'rq0-01': 'ImageNet ResNet34 (rujukan)',
    'rq1-01': 'Scratch ResNet34',
    'rq1-02': 'DINO ResNet34',
    'rq1-03': 'DINO + ChronoCon ResNet34',
    'rq1-04': 'ImageNet ViT-S (LLRD)',
    'rq1-05': 'MAE ViT-S (LLRD)',
    'rq1-06': 'DINO ViT-S (LLRD)',
    'rq1-07': 'ImageNet → bone age RSNA ResNet34',
    'rq1-08': 'SimCLR ResNet34',
    'rq1-10': 'BYOL ResNet34',
    'rq1-12': 'SupCon + CE ResNet34',
    'rq1-13': 'SupCon berbobot jarak grade ResNet34 (modifikasi)',
    'rq1-14': 'SSL normalitas grade 0 ResNet34',
    'rq1-15': 'Gate normalitas 0 vs ≥1 ResNet34 (modifikasi)',
    'rq1-17a': 'Linear probe ImageNet ResNet34',
    'rq1-17b': 'LP-FT ImageNet ResNet34',
    'rq1-17c': 'Linear probe SSL terbaik',
    'rq1-17d': 'LP-FT SSL terbaik',
    'rq1-18a': 'SKELEX ViT-L linear probe',
    'rq1-18b': 'SKELEX ViT-L fine-tune (LLRD)',
}
SELECTION_LABELS = {BEST: 'metode terbaik yang sudah ada', BEST_NO_SKELEX: 'metode terbaik selain SKELEX',
                    BEST_SSL: 'SSL terbaik'}
# Candidates fixed in the prespecification: priority-A existing methods with full fine-tuning.
BEST_CANDIDATES = ('rq0-01', 'rq1-01', 'rq1-02', 'rq1-03', 'rq1-04', 'rq1-05', 'rq1-06', 'rq1-07', 'rq1-08', 'rq1-10',
                   'rq1-12', 'rq1-14', 'rq1-18b')
SSL_CANDIDATES = ('rq1-02', 'rq1-03', 'rq1-05', 'rq1-06', 'rq1-08', 'rq1-10')
FAMILIES = {
    'F1': [(a, 'rq0-01') for a in ('rq1-01', 'rq1-02', 'rq1-07', 'rq1-08', 'rq1-10', 'rq1-12', 'rq1-14', 'rq1-18b')],
    'F2': [('rq1-03', 'rq1-02'), ('rq1-04', 'rq0-01'), ('rq1-05', 'rq1-04'), ('rq1-06', 'rq1-04'),
           ('rq1-08', 'rq1-02'), ('rq1-10', 'rq1-02'), ('rq1-18b', BEST_NO_SKELEX)],
    'F3': [('rq1-13', 'rq1-12'), ('rq1-13', BEST), ('rq1-15', 'rq1-14'), ('rq1-15', BEST)],
    'F4': [('rq1-17a', 'rq0-01'), ('rq1-17b', 'rq0-01'), ('rq1-17c', BEST_SSL), ('rq1-17d', BEST_SSL),
           ('rq1-18a', 'rq1-18b')],
}
FAMILY_TITLES = {
    'F1': 'F1: pre-training / SSL vs rujukan ImageNet ResNet34',
    'F2': 'F2: antar-metode dan arsitektur',
    'F3': 'F3: modifikasi (vs versi asli dan vs metode terbaik yang sudah ada)',
    'F4': 'F4: strategi fine-tuning',
}
NOTES = {('rq1-04', 'rq0-01'): 'tercampur resep: ViT-S memakai LLRD + warmup'}


def arm_glob(root, arm, frac):
    sub = 'rq0' if arm.startswith('rq0') else 'rq1'
    return str(Path(root) / sub / f'{arm}_*_b2_f{frac}_d*_s*')


def verdict(stats):
    """Locked rules: a difference needs Holm p < .05, and it is meaningful from an absolute 0.05."""
    if stats['p_holm'] >= ALPHA:
        return 'tidak terbukti'
    return 'bermakna' if abs(stats['diff']) >= MIN_DIFF else 'kecil'


def run_metrics(pattern):
    paths = sorted(Path(p).parent / 'metrics.json' for p in load(pattern).values())
    return [json.loads(p.read_text()) for p in paths if p.exists()]


def run_summary(pattern):
    """Mean validation BACC4/QWK, test BACC4 and the selected epoch over the runs that have metrics.json."""
    ms = run_metrics(pattern)
    if not ms:
        return None
    val = np.array([m['val']['bacc4'] for m in ms])
    test = np.array([m['test']['bacc4'] for m in ms])
    return {'runs': len(ms), 'val_bacc4': float(val.mean()), 'val_qwk': float(np.mean([m['val']['qwk'] for m in ms])),
            'test_bacc4': float(test.mean()), 'test_minus_val': float((test - val).mean()),
            'best_epoch': float(np.mean([m['best_epoch'] for m in ms]))}


def select(root, candidates, fracs, expected=3):
    """Arm with the highest mean validation BACC4 over its runs at the given fractions (ties: earlier candidate).
    Provisional while a candidate has fewer than the expected runs per fraction."""
    scores, missing = {}, []
    for arm in candidates:
        vals = [m['val']['bacc4'] for f in fracs for m in run_metrics(arm_glob(root, arm, f))]
        if len(vals) < expected * len(fracs):
            missing.append(arm)
        if vals:
            scores[arm] = float(np.mean(vals))
    best = max(scores, key=lambda a: (round(scores[a], 3), -candidates.index(a))) if scores else None
    return {'arm': best, 'scores': scores, 'missing': missing, 'provisional': bool(missing) or best is None}


def selections(root):
    out = {}
    for frac in FRACTIONS:
        out[f'{BEST}_{frac}'] = select(root, BEST_CANDIDATES, (frac,))
        out[f'{BEST_NO_SKELEX}_{frac}'] = select(root, tuple(a for a in BEST_CANDIDATES if a != 'rq1-18b'), (frac,))
    ssl = select(root, SSL_CANDIDATES, FRACTIONS)
    for frac in FRACTIONS:
        out[f'{BEST_SSL}_{frac}'] = ssl
    return out


def resolve(base, frac, sel):
    """Concrete arm for a base that may be a selection, or (None, reason) while the selection is provisional."""
    if base not in SELECTIONS:
        return base, None
    s = sel[f'{base}_{frac}']
    if s['provisional']:
        return None, 'pilihan validasi belum final; kandidat belum selesai: ' + ', '.join(s['missing'])
    return s['arm'], None


def family(root, pairs, sel, n_boot):
    rows = []
    for variant, base in pairs:
        for frac in FRACTIONS:
            arm, why = resolve(base, frac, sel)
            row = {'arm': variant, 'base_key': base, 'base': arm, 'frac': frac, 'note': NOTES.get((variant, base), '')}
            pair = paired(arm_glob(root, variant, frac), arm_glob(root, arm, frac)) if arm else None
            if arm is not None and arm == variant:
                row.update(status='tidak dijalankan', reason='arm terpilih sama dengan arm yang diuji', bacc4={'p': 1.0})
            elif pair is None:
                row.update(status='tidak dijalankan', reason=why or 'arm belum dijalankan', bacc4={'p': 1.0})
            else:
                row.update(status='dihitung', **contrast(*pair, n_boot, icc=True))
            rows.append(row)
    add_holm(rows)
    for r in rows:
        r['verdict'] = verdict(r['bacc4']) if r['status'] == 'dihitung' else r['status']
    return rows


def analyse(root, n_boot=10000):
    root = Path(root)
    sel = selections(root)
    families = {k: family(root, pairs, sel, n_boot) for k, pairs in FAMILIES.items()}
    predictions, summaries = [], []
    for frac in FRACTIONS:
        for arm in LABELS:
            frames = [pd.read_csv(p) for p in load(arm_glob(root, arm, frac)).values()]
            if frames:
                predictions.append({'frac': frac, 'arm': arm, **prediction_stats(frames)})
            s = run_summary(arm_glob(root, arm, frac))
            if s:
                summaries.append({'frac': frac, 'arm': arm, **s})
    return {'selections': sel, 'families': families, 'predictions': predictions, 'summaries': summaries}


def gap(root, device='cuda', workers=4):
    """e8.py gap over every RQ0-01 and RQ1 checkpoint (tuning runs excluded)."""
    from .e8 import gap as e8_gap

    root = Path(root)
    runs = [p.parent for p in sorted(root.glob('rq0/rq0-01_*/model_best.pt')) + sorted(root.glob('rq1/rq1-*/model_best.pt'))
            if 'tune' not in p.parent.name]
    rows = e8_gap(root, device, workers, runs=runs)
    for r in rows:
        r['arm'] = r['run'].split('_')[0]
    return rows


def label(arm):
    return LABELS.get(arm) or SELECTION_LABELS.get(arm) or '—'


def fmt_contrast(r):
    frac = f"{float(r['frac']):.0%}"
    if r['status'] != 'dihitung':
        return (f"| {frac} | {label(r['arm'])} | {label(r['base'] or r['base_key'])} | — | — | — | — | — | "
                f"— | {r['verdict']} ({r['reason']}) | — | — | — | — |")
    a, q, i = r['bacc4'], r['qwk'], r['icc']
    note = f" ({r['note']})" if r['note'] else ''
    base = label(r['base']) + (f" [{label(r['base_key'])}]" if r['base_key'] in SELECTIONS else '')
    return (f"| {frac} | {label(r['arm'])}{note} | {base} | {r['runs']} | {a['base']:.3f} | {a['variant']:.3f} | "
            f"{a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {a['p']:.3f} | {a['p_holm']:.3f} | {r['verdict']} | "
            f"{q['diff']:+.3f} | {q['p']:.3f} | {i['diff']:+.3f} | {i['p']:.3f} |")


def report(results, gaps=None):
    lines = ['# RQ1: pre-training dan SSL vs inisialisasi ImageNet', '',
             '*Post hoc*: test RAM-H1200 sudah dipakai di repo lama, sehingga hasil ini tidak dipakai untuk memilih '
             'konfigurasi tesis; semua pilihan memakai validasi. Aturan dikunci sebelum run pertama: '
             '`rules/rq1_prespecification.md`. Holm per keluarga (F1–F4) dengan ukuran tetap; kontras yang belum '
             'dijalankan masuk Holm dengan p = 1. p-value QWK dan ICC tidak dikoreksi. Putusan: bermakna = Holm p < 0,05 '
             'dan |selisih| ≥ 0,05; kecil = Holm p < 0,05 dan |selisih| < 0,05; selain itu tidak terbukti '
             '(bukan bukti kesetaraan).', '',
             'ImageNet ViT-S memakai `augreg_in1k` (ImageNet-1k), bukan in21k seperti E2/E6, jadi angkanya tidak '
             'sebanding langsung dengan E2/E6. Encoder SSL E1 melihat citra val tanpa label.', '',
             '## Pilihan berbasis validasi (rata-rata BACC4 validasi)', '']
    for key in [f'{BEST}_{f}' for f in FRACTIONS] + [f'{BEST_NO_SKELEX}_{f}' for f in FRACTIONS] + [f'{BEST_SSL}_1.0']:
        name, frac = key.rsplit('_', 1)
        s = results['selections'][key]
        title = SELECTION_LABELS[name][0].upper() + SELECTION_LABELS[name][1:] + (' (6 run, kedua fraksi)' if name == BEST_SSL else f', {float(frac):.0%}')
        state = 'sementara' if s['provisional'] else 'final'
        ranking = ', '.join(f"{label(a)} {v:.3f}" for a, v in sorted(s['scores'].items(), key=lambda kv: -kv[1]))
        missing = f" Belum selesai: {', '.join(s['missing'])}." if s['missing'] else ''
        lines.append(f"- **{title}** ({state}): {label(s['arm'])}. Urutan: {ranking}.{missing}")
    for fam, rows in results['families'].items():
        lines += ['', f'## {FAMILY_TITLES[fam]} (Holm atas {len(rows)} kontras)', '',
                  '| Label | Arm | Pembanding | Run | BACC4 pembanding | BACC4 arm | Selisih [95% CI] | p mentah | Holm p | Putusan | '
                  'Selisih QWK | p | Selisih ICC | p |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        lines += [fmt_contrast(r) for r in rows]
    lines += ['', '## Validasi, test, dan epoch terpilih (rata-rata run)', '',
              'Kolom test − val adalah cek deskriptif optimisme test, bukan uji. Validasi hanya 36 pasien.', '',
              '| Label | Arm | Run | Val BACC4 | Val QWK | Test BACC4 | Test − val | Epoch terpilih |',
              '|---|---|---|---|---|---|---|---|']
    for r in results['summaries']:
        lines.append(f"| {float(r['frac']):.0%} | {label(r['arm'])} | {r['runs']} | {r['val_bacc4']:.3f} | "
                     f"{r['val_qwk']:.3f} | {r['test_bacc4']:.3f} | {r['test_minus_val']:+.3f} | {r['best_epoch']:.1f} |")
    grades = [str(g) for g in NATIVE_SCORES]
    lines += ['', '## Recall per grade (test)', '', '| Label | Arm | Run | Proporsi prediksi grade 0 | ' +
              ' | '.join(f'Recall {g}' for g in grades) + ' |', '|---|---|---|---|' + '---|' * len(grades)]
    for r in results['predictions']:
        lines.append(f"| {float(r['frac']):.0%} | {label(r['arm'])} | {r['runs']} | {r['pred_share_grade0']:.3f} | " +
                     ' | '.join(f"{r['recall'][g]:.3f}" for g in grades) + ' |')
    if gaps:
        lines += ['', '## Celah train − val pada checkpoint terpilih (rata-rata run)', '',
                  '| Label | Arm | Run | Train BACC4 | Val BACC4 | Celah | Train QWK | Val QWK | Celah | Epoch | Loss akhir |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in gaps:
            lines.append(f"| {float(r['frac']):.0%} | {label(r['arm'])} | {r['runs']} | {r['train_bacc4']:.3f} | "
                         f"{r['val_bacc4']:.3f} | {r['gap_bacc4']:+.3f} | {r['train_qwk']:.3f} | {r['val_qwk']:.3f} | "
                         f"{r['gap_qwk']:+.3f} | {r['best_epoch']:.1f} | {r['loss_last']:.3f} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    from .e8 import summarise_gap

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', nargs='?', default='report', choices=['report', 'gap'])
    p.add_argument('--runs', default='../runs')
    p.add_argument('--out', default='../runs/rq1_analysis')
    p.add_argument('--n-boot', type=int, default=10000)
    args = p.parse_args(argv)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    if args.command == 'gap':
        (dest / 'gap.json').write_text(json.dumps(gap(args.runs), indent=2))
        return
    results = analyse(args.runs, args.n_boot)
    path = dest / 'gap.json'
    gaps = summarise_gap(json.loads(path.read_text())) if path.exists() else None
    text = report(results, gaps)
    (dest / 'report.md').write_text(text)
    (dest / 'rq1.json').write_text(json.dumps({**results, 'gap': gaps}, indent=2))
    print(text)


if __name__ == '__main__':
    main()
