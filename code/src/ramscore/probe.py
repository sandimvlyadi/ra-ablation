"""Frozen-encoder probes of pre-trained backbones on BE grading (E1 analysis).

Each encoder embeds every train/val/test ROI once (no augmentation, pooled
features), then two probes are fitted on the train joints and scored on test:
  knn     cosine k-NN vote (k = 20, weight exp(sim / 0.07), as in DINO), each
          neighbour's vote divided by its grade's train frequency (balanced)
  linear  multinomial logistic regression on standardised features,
          class_weight='balanced', C chosen on validation BACC4
Both run on all train patients and on 10% patient subsets (3 draws, stratified
as in E2). Collapse diagnostics: mean per-dimension SD of the L2-normalised
features (about 1/sqrt(d) when spread, -> 0 when collapsed) and the effective
rank of their covariance (Roy & Vetterli, 2007).

Test predictions go to <out>/pred/<encoder>/<probe>_<subset>/test_predictions.csv,
the train.py layout, so encoders can be compared with ramscore.compare, e.g.
  --pattern 'runs/e1_probe/pred/{}/linear_f0.1_d*'   (the 3 draws act as seeds)

Encoders: a checkpoint path (E1 encoder.pt, or a train.py model_best.pt whose
backbone is read from the run's args.json), an E1 run directory (up to 4 evenly
spaced encoder_ep<N>.pt, to follow the encoder over pre-training),
'random:<timm name>' (A0 init) or 'imagenet:<timm name>' (A1).

Usage: python -m ramscore.probe runs/e1_dino_resnet34/encoder.pt random:resnet34 imagenet:resnet34 \
    runs/b0_resnet34_f1.0_s42/model_best.pt --out runs/e1_probe
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .data import NATIVE_SCORES, BEDataset, load_records, subsample_patients
from .metrics import bootstrap_ci, compute_metrics
from .models import ALIASES
from .train import DEFAULT_ROOT

KEYS = ('bacc4', 'bacc5', 'qwk', 'macro_f1', 'recall_ge2')


def expand(spec, n=4):
    """A run directory -> up to n evenly spaced encoder_ep<N>.pt checkpoints (the last always kept)."""
    path = Path(spec)
    if not path.is_dir():
        return [spec]
    ckpts = sorted(path.glob('encoder_ep*.pt'), key=lambda p: int(p.stem.removeprefix('encoder_ep')))
    if not ckpts:
        return [str(path / 'encoder.pt')]
    k = min(n, len(ckpts))
    return [str(ckpts[round((i + 1) * len(ckpts) / k) - 1]) for i in range(k)]


def build_encoder(spec):
    """(timm backbone with num_classes=0, short name) for a checkpoint path or random:/imagenet: spec."""
    kind, _, rest = spec.partition(':')
    if kind in ('random', 'imagenet') and rest:
        net = timm.create_model(rest, pretrained=kind == 'imagenet', num_classes=0, in_chans=1)
        return net, f'{kind}_{rest.split(".")[0]}'
    path = Path(spec)
    state = torch.load(path, map_location='cpu')
    name = state.get('backbone')
    if name is None:
        model = json.loads((path.parent / 'args.json').read_text())['model']
        name = ALIASES.get(model, model)
    net = timm.create_model(name, num_classes=0, in_chans=1)
    prefix = timm.create_model(name, pretrained=False).pretrained_cfg['classifier'] + '.'
    net.load_state_dict({k: v for k, v in state.get('model', state).items() if not k.startswith(prefix)})
    epoch = f"_ep{state['epoch'] + 1}" if path.stem.startswith('encoder') and 'epoch' in state else ''
    return net, f'{path.parent.name}{epoch}'


@torch.no_grad()
def embed(net, ds, device, batch_size=256, workers=4):
    net.eval().to(device)
    out = []
    for batch in DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=True):
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            out.append(net(batch['img'].to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def collapse_stats(feats):
    z = F.normalize(torch.as_tensor(feats, dtype=torch.float64), dim=1)
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s ** 2 / (s ** 2).sum()
    return {'dim': int(z.shape[1]), 'feature_sd': float(z.std(0).mean()),
            'sd_if_uniform': float(1 / np.sqrt(z.shape[1])),
            'effective_rank': float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))}


def knn_predict(train_x, train_y, test_x, num_classes, k=20, temperature=0.07, chunk=2048):
    """Class indices from a balanced, similarity-weighted cosine k-NN vote."""
    tr = F.normalize(torch.as_tensor(train_x, dtype=torch.float32), dim=1)
    te = F.normalize(torch.as_tensor(test_x, dtype=torch.float32), dim=1)
    y = torch.as_tensor(train_y)
    inv_freq = 1.0 / torch.bincount(y, minlength=num_classes).clamp_min(1).float()
    preds = []
    for i in range(0, len(te), chunk):
        sim, idx = (te[i:i + chunk] @ tr.T).topk(min(k, len(tr)), dim=1)
        votes = torch.zeros(len(idx), num_classes)
        votes.scatter_add_(1, y[idx], (sim / temperature).exp() * inv_freq[y[idx]])
        preds.append(votes.argmax(1))
    return torch.cat(preds).numpy()


def linear_predict(train_x, train_y, val_x, val_y, test_x, cs=(0.01, 0.1, 1.0)):
    """Class indices from balanced logistic regression, C picked on validation BACC4; returns (pred, C)."""
    scaler = StandardScaler().fit(train_x)
    tr, va, te = scaler.transform(train_x), scaler.transform(val_x), scaler.transform(test_x)
    scores = np.asarray(NATIVE_SCORES)
    best = None
    for c in cs:
        clf = LogisticRegression(C=c, class_weight='balanced', max_iter=3000).fit(tr, train_y)
        bacc4 = compute_metrics(scores[val_y], scores[clf.predict(va)])['bacc4']
        if best is None or bacc4 > best[0]:
            best = (bacc4, c, clf)
    return best[2].predict(te), best[1]


def probe(feats, labels, test_patients, train_subsets, n_boot=1000):
    """(metrics, test predictions on native scores) of both probes for every train subset
    (dict name -> train indices); predictions are keyed '<probe>_<subset>'."""
    scores = np.asarray(NATIVE_SCORES)
    gt = scores[labels['test']]
    out, preds = {}, {}
    for subset, idx in train_subsets.items():
        x, y = feats['train'][idx], labels['train'][idx]
        knn = knn_predict(x, y, feats['test'], len(scores))
        lin, c = linear_predict(x, y, feats['val'], labels['val'], feats['test'])
        out[subset] = {}
        for probe_name, pred in (('knn', knn), ('linear', lin)):
            preds[f'{probe_name}_{subset}'] = scores[pred]
            m = compute_metrics(gt, scores[pred])
            res = {k: m[k] for k in KEYS} | {'recall_per_grade': m['recall_per_grade']}
            if n_boot and subset == 'f1.0':
                df = pd.DataFrame({'patient': test_patients, 'gt': gt, 'pred': scores[pred]})
                res['ci95'] = bootstrap_ci(df, keys=('bacc4', 'qwk'), n_boot=n_boot)
            if probe_name == 'linear':
                res['C'] = c
            out[subset][probe_name] = res
    return out, preds


def summarize(results):
    lines = ['| Encoder | kNN BACC4 100% | Linear BACC4 100% [CI 95%] | Linear BACC4 10% (mean of draws) '
             '| Linear QWK 100% | Feature SD (uniform) | Eff. rank |', '|---|---|---|---|---|---|---|']
    for name, r in results.items():
        p, c = r['probes'], r['collapse']
        f10 = [v['linear']['bacc4'] for k, v in p.items() if k != 'f1.0']
        ci = p['f1.0']['linear'].get('ci95', {}).get('bacc4', [np.nan, np.nan])
        lines.append(f"| {name} | {p['f1.0']['knn']['bacc4']:.3f} | {p['f1.0']['linear']['bacc4']:.3f} "
                     f"[{ci[0]:.3f}, {ci[1]:.3f}] | {np.mean(f10) if f10 else np.nan:.3f} | "
                     f"{p['f1.0']['linear']['qwk']:.3f} | {c['feature_sd']:.4f} ({c['sd_if_uniform']:.4f}) | "
                     f"{c['effective_rank']:.1f} |")
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('encoders', nargs='+')
    p.add_argument('--data-root', default=DEFAULT_ROOT)
    p.add_argument('--out', default='runs/e1_probe')
    p.add_argument('--fraction', type=float, default=0.1)
    p.add_argument('--fraction-seeds', type=int, nargs='*', default=[0, 1, 2])
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--limit', type=int, default=0, help='First N records per split (smoke test)')
    args = p.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records = {s: load_records(args.data_root, s) for s in ('train', 'val', 'test')}
    if args.limit:
        records = {s: r[:args.limit] for s, r in records.items()}
    t0 = time.time()
    ds = {s: BEDataset(r, NATIVE_SCORES) for s, r in records.items()}
    labels = {s: d.labels for s, d in ds.items()}
    print(f'Loaded ROIs in {time.time() - t0:.0f}s: ' + ', '.join(f'{s} {len(d)}' for s, d in ds.items()))
    position = {id(r): i for i, r in enumerate(records['train'])}
    subsets = {'f1.0': np.arange(len(records['train']))}
    for seed in args.fraction_seeds:
        keep = subsample_patients(records['train'], args.fraction, seed)
        subsets[f'f{args.fraction}_d{seed}'] = np.array([position[id(r)] for r in keep])
    test_patients = [r['patient'] for r in records['test']]
    test_key = pd.DataFrame({k: [r[k] for r in records['test']] for k in ('case', 'joint', 'patient')})
    test_key['gt'] = np.asarray(NATIVE_SCORES)[labels['test']]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in [s for e in args.encoders for s in expand(e)]:
        t0 = time.time()
        net, name = build_encoder(spec)
        feats = {s: embed(net, d, device) for s, d in ds.items()}
        metrics, preds = probe(feats, labels, test_patients, subsets, args.n_boot)
        results[name] = {'spec': spec, 'collapse': collapse_stats(feats['train']), 'probes': metrics}
        (out / f'{name}.json').write_text(json.dumps(results[name], indent=2))
        for key, pred in preds.items():   # same layout as train.py runs, for ramscore.compare
            run = out / 'pred' / name / key
            run.mkdir(parents=True, exist_ok=True)
            test_key.assign(pred=pred).to_csv(run / 'test_predictions.csv', index=False)
        r = results[name]['probes']['f1.0']
        print(f"{name}: knn bacc4 {r['knn']['bacc4']:.3f}, linear bacc4 {r['linear']['bacc4']:.3f} "
              f"({time.time() - t0:.0f}s)")
        del net
        torch.cuda.empty_cache()
    print(summarize(results))


if __name__ == '__main__':
    main()
