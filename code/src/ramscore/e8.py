"""E8: ablation of the training augmentation on ResNet34 + A1 ImageNet + B2, at 100% and 10% labels.

  base       ShiftScaleRotate + RandomBrightnessContrast, the recipe of every earlier classifier run
  geom       ShiftScaleRotate only
  intensity  RandomBrightnessContrast only
  none       no augmentation

Post hoc: the test set was already used. The contrasts, the Holm family and the verdict rules were fixed before the
first run in papers/e8_prespecification_2026-11-08.md. Paired patient bootstrap of compare.py (hier: runs redrawn).

Usage: uv run --group analysis python -m ramscore.e8 gap      # train and validation scores of each checkpoint; GPU
       uv run --group analysis python -m ramscore.e8          # contrasts and report; reads gap.json when it exists
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .data import NATIVE_SCORES, get_transform
from .e6abl import add_holm, contrast, load, paired
from .e7 import prediction_stats

ARMS = ('none', 'geom', 'intensity')
FRACTIONS = ('1.0', '0.1')
TAIL = 'a1_resnet34_b2_f{frac}_d*_s*'
MIN_DIFF = 0.05
ALPHA = 0.05


def arm_glob(root, arm, frac):
    return str(Path(root) / 'e8' / f'e8_{arm}_{TAIL.format(frac=frac)}')


def e3_glob(root, frac):
    return str(Path(root) / 'e3' / f'e3_{TAIL.format(frac=frac)}')


def verdict(stats):
    """Locked rules: a difference needs Holm p < .05, and it is meaningful from an absolute 0.05."""
    if stats['p_holm'] >= ALPHA:
        return 'not shown'
    return 'meaningful' if abs(stats['diff']) >= MIN_DIFF else 'small'


def analyse(root, n_boot=10000):
    root = Path(root)
    primary, supporting, predictions = [], [], []
    for frac in FRACTIONS:
        for arm in ARMS:
            pair = paired(arm_glob(root, arm, frac), arm_glob(root, 'base', frac))
            if pair:
                primary.append({'frac': frac, 'arm': arm, **contrast(*pair, n_boot, icc=True)})
        pair = paired(arm_glob(root, 'base', frac), e3_glob(root, frac))
        if pair:
            supporting.append({'frac': frac, **contrast(*pair, n_boot, icc=True)})
        for arm in ('base', *ARMS):
            frames = [pd.read_csv(p) for p in load(arm_glob(root, arm, frac)).values()]
            if frames:
                predictions.append({'frac': frac, 'arm': arm, **prediction_stats(frames)})
    add_holm(primary)
    for r in primary:
        r['verdict'] = verdict(r['bacc4'])
    return {'primary': primary, 'supporting': supporting, 'predictions': predictions}


def gap(root, device='cuda', workers=4, runs=None):
    """BACC4 and QWK of the selected checkpoint on the unaugmented train and validation joints of its own design.
    runs: run folders to score (default: every E8 run under root/e8)."""
    import torch
    from torch.utils.data import DataLoader

    from .data import BEDataset, load_records, subsample_patients
    from .metrics import compute_metrics
    from .models import build_model
    from .train import predict

    rows = []
    if runs is None:
        runs = [p.parent for p in sorted(Path(root, 'e8').glob('e8_*/model_best.pt'))]
    for run in runs:
        args = json.loads((run / 'args.json').read_text())
        records = {s: load_records(args['data_root'], s) for s in ('train', 'val')}
        records['train'] = subsample_patients(records['train'], args['label_fraction'], args['fraction_seed'])
        net = build_model(args['model'], len(NATIVE_SCORES), args['head'])
        state = torch.load(run / 'model_best.pt', map_location='cpu')
        net.load_state_dict(state['model'])
        net.to(device)
        row = {'run': run.name, 'arm': args['aug'], 'frac': str(args['label_fraction']), 'best_epoch': state['epoch']}
        with torch.inference_mode():
            for split, recs in records.items():
                ds = BEDataset(recs, NATIVE_SCORES, args['image_size'], get_transform(split, 'none'))
                loader = DataLoader(ds, batch_size=64, num_workers=workers)
                df = predict(net, loader, args['head'], device, NATIVE_SCORES)
                m = compute_metrics(df['gt'], df['pred'])
                row[f'{split}_bacc4'], row[f'{split}_qwk'] = m['bacc4'], m['qwk']
        history = pd.read_csv(run / 'history.csv')
        row['loss_last'] = float(history['train_loss'].iloc[-1])
        rows.append(row)
    return rows


def summarise_gap(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df['gap_bacc4'] = df['train_bacc4'] - df['val_bacc4']
    df['gap_qwk'] = df['train_qwk'] - df['val_qwk']
    cols = ['train_bacc4', 'val_bacc4', 'gap_bacc4', 'train_qwk', 'val_qwk', 'gap_qwk', 'best_epoch', 'loss_last']
    out = df.groupby(['frac', 'arm'])[cols].mean().reset_index()
    out['runs'] = df.groupby(['frac', 'arm']).size().values
    return out.to_dict('records')


def report(results, gaps=None):
    lines = ['# E8: ablation of the training augmentation (ResNet34 + A1 ImageNet + B2)', '',
             'Post hoc: the test set was already used, so these results cannot pick a configuration for the thesis. '
             'Rules fixed before the first run: `papers/e8_prespecification_2026-11-08.md`. '
             'Holm over the 6 primary contrasts on BACC4; every other p-value is unadjusted.', '',
             '## Primary contrasts against `base`', '',
             '| Labels | Arm | Runs | BACC4 base | BACC4 arm | Diff [95% CI] | Holm p | Verdict | QWK diff | p | ICC diff | p |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results['primary']:
        a, q, i = r['bacc4'], r['qwk'], r['icc']
        lines.append(f"| {float(r['frac']):.0%} | `{r['arm']}` | {r['runs']} | {a['base']:.3f} | {a['variant']:.3f} | "
                     f"{a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {a['p_holm']:.3f} | {r['verdict']} | "
                     f"{q['diff']:+.3f} | {q['p']:.3f} | {i['diff']:+.3f} | {i['p']:.3f} |")
    lines += ['', '## Repeat of the same recipe: `base` (seeded) against the E3 B2 run of the same draw and seed', '',
              '| Labels | Runs | BACC4 E3 | BACC4 E8 | Diff [95% CI] | p | QWK diff | p |', '|---|---|---|---|---|---|---|---|']
    for r in results['supporting']:
        a, q = r['bacc4'], r['qwk']
        lines.append(f"| {float(r['frac']):.0%} | {r['runs']} | {a['base']:.3f} | {a['variant']:.3f} | "
                     f"{a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {a['p']:.3f} | {q['diff']:+.3f} | {q['p']:.3f} |")
    grades = [str(g) for g in NATIVE_SCORES]
    lines += ['', '## Recall per grade', '', '| Labels | Arm | Runs | Grade 0 share, predicted | ' +
              ' | '.join(f'Recall {g}' for g in grades) + ' |', '|---|---|---|---|' + '---|' * len(grades)]
    for r in results['predictions']:
        lines.append(f"| {float(r['frac']):.0%} | `{r['arm']}` | {r['runs']} | {r['pred_share_grade0']:.3f} | " +
                     ' | '.join(f"{r['recall'][g]:.3f}" for g in grades) + ' |')
    if gaps:
        lines += ['', '## Train-validation gap of the selected checkpoint (unaugmented joints, mean over runs)', '',
                  '| Labels | Arm | Runs | Train BACC4 | Val BACC4 | Gap | Train QWK | Val QWK | Gap | Best epoch | Last train loss |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in gaps:
            lines.append(f"| {float(r['frac']):.0%} | `{r['arm']}` | {r['runs']} | {r['train_bacc4']:.3f} | {r['val_bacc4']:.3f} | "
                         f"{r['gap_bacc4']:+.3f} | {r['train_qwk']:.3f} | {r['val_qwk']:.3f} | {r['gap_qwk']:+.3f} | "
                         f"{r['best_epoch']:.1f} | {r['loss_last']:.3f} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', nargs='?', default='report', choices=['report', 'gap'])
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='runs/e8_analysis')
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
    (dest / 'e8.json').write_text(json.dumps({**results, 'gap': gaps}, indent=2))
    print(text)


if __name__ == '__main__':
    main()
