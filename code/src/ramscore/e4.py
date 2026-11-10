"""E4: cross-dataset transfer to RheumaVIT (proposal RQ3 / H3, 5.3 and 6).

  direct   every source run of the 4 configurations predicts the RheumaVIT ROIs (rheumavit.py) without any
           training -> <out>/direct/<run>/predictions.csv (all 330 patients; the test set is e4_split 'test')
  fewshot  fine-tune each source run on 5 or 20 RheumaVIT patients of the pool and predict the test set ->
           <out>/fewshot/k<k>_d<draw>/<run>/predictions.csv. 3 nested draws: 20 patients by systematic sampling
           along (worst grade, random key), so every worst-grade stratum is represented proportionally, and the
           5 the same way from those 20. Fixed recipe set before any few-shot result (no target validation):
           the whole model incl. its B2 head, AdamW lr 1e-5 (1/10 of the source), cosine to 1%, 500 steps,
           batch 32, AMP, the source augmentation, source seed; logit adjustment uses the few-shot priors.
           Training grade 4 (absent from the 5 output classes) is taken as 3. K1 uses its d0_s42, d1_s1337 and
           d2_s2026 runs, so every configuration has 3 source runs x 3 draws x 2 sizes = 18 runs (72 in all).
  analyse  (also, once fewshot has run) external metrics per configuration x patients (0 = direct, 5, 20) over
           the same 3 source runs, and the secondary contrasts K3 - K2 and K3 - K4 at each size (Holm over the 2
           within a size; exploratory).
           4-level metrics (0, 1, 2, >= 3) on the 13 joints shared by both datasets, internal (RAM-H1200 test,
           the run's own test_predictions.csv) and external (RheumaVIT test); drop = internal - external.
           Primary H3 contrasts: drop(A0) - drop(A4) and drop(A1) - drop(A4) at 100% labels, B2 (positive =
           SSL drops less), Holm over the 2. RAM-H1200 and RheumaVIT patients are resampled independently,
           with the same replicates for every configuration (paired); 'hier' also redraws the seed runs.
  anchor   inter-reader anchor (proposal 6): each RheumaVIT reader against the lower median of the other two
           (or the one other reader who scored the joint), and every model against that same reference on the
           same joints; metrics averaged over the 3 readers; model - reader with a patient bootstrap (hier over
           runs). Scope 'test' (90 patients: direct and 20-patient few-shot) and 'all' (330 patients, direct
           only: no direct-transfer model saw RheumaVIT).

Configurations (selected on validation BACC4, proposal 7.5; B fixed at B2 so H3 compares encoders only):
  K1 A4 DINO + B2, 10% labels (9 runs)   K2 A0 scratch + B2, 100% (3)
  K3 A4 DINO + B2, 100% (3)              K4 A1 ImageNet + B2, 100% (3)
RheumaVIT grade 4 (absent from RAM-H1200) only occurs in the reference and maps to level 3 like 3 and 5.

Usage: python -m ramscore.e4 direct && uv run --group analysis python -m ramscore.e4 analyse
"""
import argparse
import glob
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .compare import RunPicks, flat_levels, holm, metrics_from_confusion, patient_confusions, run_levels, two_sided_p
from .data import LEVEL4, NATIVE_SCORES, BEDataset, get_transform
from .models import Criterion, build_model
from .rheumavit import JOINTS, READERS, load_records
from .train import predict, seed_everything

CONFIGS = {
    'K1': 'runs/e3/e3_a4_resnet34_b2_f0.1_d*_s*',
    'K2': 'runs/e3/e3_a0_resnet34_b2_f1.0_d*_s*',
    'K3': 'runs/e3/e3_a4_resnet34_b2_f1.0_d*_s*',
    'K4': 'runs/e3/e3_a1_resnet34_b2_f1.0_d*_s*',
}
LABELS = {'K1': 'A4 DINO + B2, 10%', 'K2': 'A0 scratch + B2, 100%', 'K3': 'A4 DINO + B2, 100%',
          'K4': 'A1 ImageNet + B2, 100%'}
PRIMARY = (('K2', 'K3'), ('K4', 'K3'))
SHARED_JOINTS = tuple(JOINTS.values())
LEVELS = (0, 1, 2, 3)
KEYS = ('bacc4', 'qwk', 'macro_f1', 'recall_ge2', 'binary_sensitivity')
ALL_SCORES = (0, 1, 2, 3, 4, 5)
SIZES = (5, 20)
DRAWS = (0, 1, 2)
K1_SOURCES = ('_d0_s42', '_d1_s1337', '_d2_s2026')
RECIPE = {'steps': 500, 'lr': 1e-5, 'weight_decay': 1e-3, 'batch_size': 32}


def source_runs(config):
    return sorted(Path(p).parent for p in glob.glob(f'{CONFIGS[config]}/model_best.pt'))


def fewshot_sources(config):
    runs = source_runs(config)
    return [r for r in runs if r.name.endswith(K1_SOURCES)] if len(runs) > 3 else runs


def fewshot_patients(manifest, draw):
    """{20: patients, 5: patients} of the pool; the 5 are a subset of the 20, both spread over worst grades."""
    m = pd.read_csv(manifest).query("e4_split == 'pool'")
    worst = m.groupby('patient')['score'].max().map(LEVEL4)
    rng = np.random.default_rng(1000 + draw)

    def systematic(patients, k):
        key = rng.random(len(patients))
        order = sorted(range(len(patients)), key=lambda i: (worst[patients[i]], key[i]))
        step = len(patients) / k
        start = rng.random() * step
        return [patients[order[int(start + i * step)]] for i in range(k)]

    p20 = systematic(sorted(worst.index), 20)
    return {20: p20, 5: systematic(p20, 5)}


def finetune(run, patients, manifest, test_loader, device, workers=4):
    args = json.loads((run / 'args.json').read_text())
    seed_everything(args['seed'])
    records = [dict(r, score=3 if r['score'] == 4 else r['score'])
               for r in load_records(manifest, 'pool') if r['patient'] in set(patients)]
    ds = BEDataset(records, NATIVE_SCORES, transform=get_transform('train'))
    g = torch.Generator().manual_seed(args['seed'])
    loader = DataLoader(ds, batch_size=RECIPE['batch_size'], shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=True, generator=g)
    net = build_model(args['model'], len(NATIVE_SCORES), args['head'])
    net.load_state_dict(torch.load(run / 'model_best.pt', map_location='cpu')['model'])
    net.to(device)
    criterion = Criterion(args['head'], ds.labels, len(NATIVE_SCORES), args['class_weight'],
                          args['logit_adjust']).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, RECIPE['steps'], eta_min=RECIPE['lr'] * 0.01)
    scaler = torch.amp.GradScaler()
    net.train()
    step, losses = 0, []
    while step < RECIPE['steps']:
        for batch in loader:
            img, target = batch['img'].to(device, non_blocking=True), batch['label'].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type):
                loss = criterion(net(img).float(), target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step == RECIPE['steps']:
                break
    return predict(net, test_loader, args['head'], device, NATIVE_SCORES), len(ds), float(np.mean(losses[-50:]))


def fewshot(manifest, out):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = DataLoader(BEDataset(load_records(manifest, 'test'), ALL_SCORES), batch_size=128, num_workers=4)
    draws = {d: fewshot_patients(manifest, d) for d in DRAWS}
    jobs = [(c, run, k, d) for c in CONFIGS for k in SIZES for d in DRAWS for run in fewshot_sources(c)]
    for i, (config, run, k, d) in enumerate(jobs, 1):
        dest = Path(out) / 'fewshot' / f'k{k}_d{d}' / run.name
        if (dest / 'predictions.csv').exists():
            continue
        t0 = time.time()
        df, n_joints, loss = finetune(run, draws[d][k], manifest, test_loader, device)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'args.json').write_text(json.dumps({'config': config, 'source': str(run), 'k': k, 'draw': d,
                                                    'patients': draws[d][k], 'train_joints': n_joints,
                                                    'final_loss': loss, **RECIPE}, indent=2))
        df.to_csv(dest / 'predictions.csv', index=False)
        print(f'[{i}/{len(jobs)}] {config} k{k} d{d} {run.name}: {n_joints} joints, loss {loss:.3f}, '
              f'{time.time() - t0:.0f}s', flush=True)


def direct(manifest, out, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = BEDataset(load_records(manifest), ALL_SCORES)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4)
    for config in CONFIGS:
        for run in source_runs(config):
            dest = Path(out) / 'direct' / run.name / 'predictions.csv'
            if dest.exists():
                continue
            args = json.loads((run / 'args.json').read_text())
            net = build_model(args['model'], len(NATIVE_SCORES), args['head'])
            net.load_state_dict(torch.load(run / 'model_best.pt', map_location='cpu')['model'])
            df = predict(net.to(device), loader, args['head'], device, NATIVE_SCORES)
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dest, index=False)
            print(f'{config} {run.name}: {len(df)} joints')


def to_levels(df):
    """Shared joints only, grades merged to the 4 levels; columns patient, gt, pred."""
    df = df[df['joint'].isin(SHARED_JOINTS)]
    return pd.DataFrame({'patient': df['patient'].to_numpy(), 'gt': df['gt'].map(LEVEL4).to_numpy(),
                         'pred': df['pred'].map(LEVEL4).to_numpy()})


def load_predictions(out, manifest):
    """{config: [(internal df, external df) per source run]}, both at the 4 levels."""
    test = set(pd.read_csv(manifest).query("e4_split == 'test'")['patient'])
    runs = {}
    for config in CONFIGS:
        runs[config] = []
        for run in source_runs(config):
            ext = pd.read_csv(Path(out) / 'direct' / run.name / 'predictions.csv')
            internal = to_levels(pd.read_csv(run / 'test_predictions.csv'))
            runs[config].append((internal, to_levels(ext[ext['patient'].isin(test)])))
    return runs


def confusions(dfs):
    patients = sorted(dfs[0]['patient'].unique())
    return np.stack([patient_confusions(df, patients, LEVELS) for df in dfs])


def fewshot_levels(n_runs):
    """(few-shot draw, source run) design of a few-shot cell, whose runs are ordered by draw, then source."""
    per_draw = n_runs // len(DRAWS)
    return [(d, i) for d in DRAWS for i in range(per_draw)]


def analyse(runs, n_boot=10000, seed=0, alpha=0.05, levels=None):
    """Point estimates per configuration and the H3 drop contrasts with paired bootstrap CIs.

    levels: {config: (draw, seed) per source run}. A configuration redraws the same runs internally and externally,
    and configurations on one design redraw the same runs.
    """
    rng = np.random.default_rng(seed)
    picks = RunPicks(n_boot, seed)
    cms = {c: {s: confusions([r[i] for r in rs]) for i, s in enumerate(('internal', 'external'))}
           for c, rs in runs.items()}
    n_pat = {s: next(iter(cms.values()))[s].shape[1] for s in ('internal', 'external')}
    w = {s: rng.multinomial(n, np.full(n, 1 / n), size=n_boot) for s, n in n_pat.items()}
    point, fixed, hier, per_run = {}, {}, {}, {}
    for c in cms:
        n_runs = cms[c]['internal'].shape[0]
        pick = picks((levels or {}).get(c, flat_levels(n_runs)))
        point[c], fixed[c], hier[c], per_run[c] = {}, {}, {}, {}
        for s in ('internal', 'external'):
            full = metrics_from_confusion(cms[c][s].sum(1), LEVELS)
            boot = metrics_from_confusion(np.einsum('bp,rpij->brij', w[s], cms[c][s]), LEVELS)
            per_run[c][s] = {k: full[k].tolist() for k in KEYS}
            point[c][s] = {k: float(full[k].mean()) for k in KEYS}
            fixed[c][s] = {k: boot[k].mean(1) for k in KEYS}
            hier[c][s] = {k: np.take_along_axis(boot[k], pick, 1).mean(1) for k in KEYS}
        for d in (point, fixed, hier):
            d[c]['drop'] = {k: d[c]['internal'][k] - d[c]['external'][k] for k in KEYS}
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    contrasts = []
    for a, b in PRIMARY:
        row = {'contrast': f'drop({a}) - drop({b})', 'a': a, 'b': b}
        for k in KEYS:
            d_fixed = fixed[a]['drop'][k] - fixed[b]['drop'][k]
            d_hier = hier[a]['drop'][k] - hier[b]['drop'][k]
            row.update({f'{k}_diff': point[a]['drop'][k] - point[b]['drop'][k],
                        f'{k}_ci_fixed': np.nanpercentile(d_fixed, [lo, hi]).tolist(),
                        f'{k}_ci_hier': np.nanpercentile(d_hier, [lo, hi]).tolist(),
                        f'{k}_p_hier': two_sided_p(d_hier[~np.isnan(d_hier)])})
        contrasts.append(row)
    for k in KEYS:
        for row, p in zip(contrasts, holm(np.array([r[f'{k}_p_hier'] for r in contrasts]))):
            row[f'{k}_p_holm'] = float(p)
    ci = {c: {s: {k: np.nanpercentile(hier[c][s][k], [lo, hi]).tolist() for k in KEYS}
              for s in ('internal', 'external', 'drop')} for c in cms}
    return point, ci, per_run, contrasts


def load_fewshot(out, manifest):
    """{(config, k): [external df per run]} at the 4 levels; k = 0 is direct transfer of the same source runs."""
    test = set(pd.read_csv(manifest).query("e4_split == 'test'")['patient'])
    cells = {}
    for c in CONFIGS:
        names = [r.name for r in fewshot_sources(c)]
        direct_dfs = [pd.read_csv(Path(out) / 'direct' / n / 'predictions.csv') for n in names]
        cells[(c, 0)] = [to_levels(df[df['patient'].isin(test)]) for df in direct_dfs]
        for k in SIZES:
            paths = [Path(out) / 'fewshot' / f'k{k}_d{d}' / n / 'predictions.csv' for d in DRAWS for n in names]
            if not all(p.exists() for p in paths):
                return None
            cells[(c, k)] = [to_levels(pd.read_csv(p)) for p in paths]
    return cells


def fewshot_analysis(cells, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    patients = sorted(next(iter(cells.values()))[0]['patient'].unique())
    w = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=n_boot)
    picks = RunPicks(n_boot, seed)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    point, sd, ci, hier = {}, {}, {}, {}
    for key, dfs in cells.items():
        cm = np.stack([patient_confusions(df, patients, LEVELS) for df in dfs])
        full = metrics_from_confusion(cm.sum(1), LEVELS)
        boot = metrics_from_confusion(np.einsum('bp,rpij->brij', w, cm), LEVELS)
        pick = picks(flat_levels(len(dfs)) if key[1] == 0 else fewshot_levels(len(dfs)))
        hier[key] = {m: np.take_along_axis(boot[m], pick, 1).mean(1) for m in KEYS}
        point[key] = {m: float(full[m].mean()) for m in KEYS}
        sd[key] = {m: float(full[m].std(ddof=1)) for m in KEYS}
        ci[key] = {m: np.nanpercentile(hier[key][m], [lo, hi]).tolist() for m in KEYS}
    contrasts = []
    for k in (0, *SIZES):
        rows = []
        for a, b in (('K3', 'K2'), ('K3', 'K4')):
            row = {'k': k, 'contrast': f'{a} - {b}'}
            for m in ('bacc4', 'qwk'):
                d = hier[(a, k)][m] - hier[(b, k)][m]
                row.update({f'{m}_diff': point[(a, k)][m] - point[(b, k)][m],
                            f'{m}_ci_hier': np.nanpercentile(d, [lo, hi]).tolist(),
                            f'{m}_p_hier': two_sided_p(d[~np.isnan(d)])})
            rows.append(row)
        for m in ('bacc4', 'qwk'):
            for row, p in zip(rows, holm(np.array([r[f'{m}_p_hier'] for r in rows]))):
                row[f'{m}_p_holm'] = float(p)
        contrasts += rows
    return point, sd, ci, contrasts


def fewshot_report(point, sd, ci, contrasts):
    lines = ['', '## Few-shot adaptation (RheumaVIT test; 3 source runs x 3 patient draws; 0 = direct)', '',
             '| Config | Patients | Runs | External BACC4 [95% CI] | SD | External QWK [95% CI] | SD |',
             '|---|---|---|---|---|---|---|']
    for c in CONFIGS:
        for k in (0, *SIZES):
            p, s, q = point[(c, k)], sd[(c, k)], ci[(c, k)]
            runs = 3 if k == 0 else 3 * len(DRAWS)
            lines.append(f"| {c} {LABELS[c]} | {k} | {runs} | {p['bacc4']:.3f} [{q['bacc4'][0]:.3f}, "
                         f"{q['bacc4'][1]:.3f}] | {s['bacc4']:.3f} | {p['qwk']:.3f} [{q['qwk'][0]:.3f}, "
                         f"{q['qwk'][1]:.3f}] | {s['qwk']:.3f} |")
    lines += ['', '### Secondary contrasts (positive = A4 DINO better; Holm over the 2 within a size)', '',
              '| Patients | Contrast | BACC4 diff [95% CI] | p (Holm) | QWK diff [95% CI] | p (Holm) |',
              '|---|---|---|---|---|---|']
    for r in contrasts:
        lines.append(f"| {r['k']} | {r['contrast']} | {r['bacc4_diff']:+.3f} [{r['bacc4_ci_hier'][0]:+.3f}, "
                     f"{r['bacc4_ci_hier'][1]:+.3f}] | {r['bacc4_p_holm']:.3f} | {r['qwk_diff']:+.3f} "
                     f"[{r['qwk_ci_hier'][0]:+.3f}, {r['qwk_ci_hier'][1]:+.3f}] | {r['qwk_p_holm']:.3f} |")
    return '\n'.join(lines) + '\n'


def report(point, ci, per_run, contrasts, n_test):
    lines = ['# E4 direct transfer: RAM-H1200 -> RheumaVIT', '',
             f"External test: RheumaVIT official val + test, {n_test['patients']} patients, {n_test['joints']} "
             'joints (13 shared joints, reference = lower median of the readers). Metrics on 4 levels (0, 1, 2, '
             '>= 3); mean over source runs, 95% CI from the hierarchical bootstrap.', '',
             '| Config | Runs | Internal BACC4 | External BACC4 | Drop | Internal QWK | External QWK | Drop QWK |',
             '|---|---|---|---|---|---|---|---|']
    f = lambda c, s, k: f"{point[c][s][k]:.3f} [{ci[c][s][k][0]:.3f}, {ci[c][s][k][1]:.3f}]"
    for c in point:
        lines.append(f"| {c} {LABELS[c]} | {len(per_run[c]['internal']['bacc4'])} | {f(c, 'internal', 'bacc4')} | "
                     f"{f(c, 'external', 'bacc4')} | {f(c, 'drop', 'bacc4')} | {f(c, 'internal', 'qwk')} | "
                     f"{f(c, 'external', 'qwk')} | {f(c, 'drop', 'qwk')} |")
    lines += ['', '## Primary H3 contrasts (100% labels; positive = A4 DINO drops less)', '',
              '| Contrast | Metric | Difference | 95% CI (hier) | p (hier) | p (Holm) |', '|---|---|---|---|---|---|']
    for r in contrasts:
        for k in ('bacc4', 'qwk'):
            lines.append(f"| {r['contrast']} | {k} | {r[f'{k}_diff']:+.3f} | [{r[f'{k}_ci_hier'][0]:+.3f}, "
                         f"{r[f'{k}_ci_hier'][1]:+.3f}] | {r[f'{k}_p_hier']:.3f} | {r[f'{k}_p_holm']:.3f} |")
    return '\n'.join(lines) + '\n'


def leave_one_out(m, reader):
    """Rows scored by `reader` and by at least one other reader, with the other readers' lower median."""
    others = [f'r{o}' for o in READERS if o != reader]
    vals = m[others].to_numpy()
    ok = m[f'r{reader}'].notna().to_numpy() & ~np.isnan(vals).all(axis=1)
    ref = [int(statistics.median_low(v[~np.isnan(v)])) for v in vals[ok]]
    return m.loc[ok, ['case', 'patient', 'joint', f'r{reader}']].assign(ref=ref)


def anchor_cells(out, manifest, scope):
    """{(source, reader): [df patient/gt/pred at 4 levels per run]}; source 'reader' = the reader himself."""
    m = pd.read_csv(manifest)
    if scope == 'test':
        m = m[m['e4_split'] == 'test']
    sources = {f'{c} direct': [Path(out) / 'direct' / r.name for r in source_runs(c)] for c in CONFIGS}
    if scope == 'test':
        sources.update({f'{c} few-shot 20': [Path(out) / 'fewshot' / f'k20_d{d}' / r.name for d in DRAWS
                                             for r in fewshot_sources(c)] for c in CONFIGS})
    preds = {d: pd.read_csv(d / 'predictions.csv').set_index(['case', 'joint'])['pred']
             for dirs in sources.values() for d in dirs}
    cells = {}
    for reader in READERS:
        loo = leave_one_out(m, reader)
        gt = loo['ref'].map(LEVEL4).to_numpy()
        key = pd.MultiIndex.from_frame(loo[['case', 'joint']])
        frame = lambda pred: pd.DataFrame({'patient': loo['patient'].to_numpy(), 'gt': gt,
                                           'pred': pd.Series(pred).map(LEVEL4).to_numpy()})
        cells[('reader', reader)] = [frame(loo[f'r{reader}'].astype(int).to_numpy())]
        for name, dirs in sources.items():
            cells[(name, reader)] = [frame(preds[d].reindex(key).to_numpy()) for d in dirs]
    return cells, sorted(m['patient'].unique())


def anchor_levels(cells):
    """Design per source: direct runs by (draw, seed), few-shot runs by (few-shot draw, source run)."""
    out = {'reader': flat_levels(1)}
    for c in CONFIGS:
        out[f'{c} direct'] = run_levels([r.name for r in source_runs(c)])
        key = (f'{c} few-shot 20', READERS[0])
        if key in cells:
            out[key[0]] = fewshot_levels(len(cells[key]))
    return out


def anchor(cells, patients, n_boot=10000, seed=0, alpha=0.05, levels=None):
    """Reader-averaged BACC4 / QWK per source and model - reader differences (paired patient bootstrap).

    A source redraws the same runs for all readers, so the reader average keeps the run-level pairing.
    """
    rng = np.random.default_rng(seed)
    picks = RunPicks(n_boot, seed)
    w = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=n_boot)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    names = list(dict.fromkeys(s for s, _ in cells))
    point, boot, per_reader = {}, {}, {}
    for name in names:
        pt, bt, pr = {k: [] for k in ('bacc4', 'qwk')}, {k: [] for k in ('bacc4', 'qwk')}, {}
        for reader in READERS:
            cm = np.stack([patient_confusions(df, patients, LEVELS) for df in cells[(name, reader)]])
            full = metrics_from_confusion(cm.sum(1), LEVELS)
            b = metrics_from_confusion(np.einsum('bp,rpij->brij', w, cm), LEVELS)
            pick = picks((levels or {}).get(name, flat_levels(cm.shape[0])))
            pr[reader] = {k: float(full[k].mean()) for k in ('bacc4', 'qwk')}
            for k in ('bacc4', 'qwk'):
                pt[k].append(full[k].mean())
                bt[k].append(np.take_along_axis(b[k], pick, 1).mean(1))
        point[name] = {k: float(np.mean(v)) for k, v in pt.items()}
        boot[name] = {k: np.nanmean(np.stack(v), 0) for k, v in bt.items()}
        per_reader[name] = pr
    ci = {n: {k: np.nanpercentile(boot[n][k], [lo, hi]).tolist() for k in ('bacc4', 'qwk')} for n in names}
    diffs = {}
    for n in names[1:]:
        diffs[n] = {}
        for k in ('bacc4', 'qwk'):
            d = boot[n][k] - boot['reader'][k]
            diffs[n][k] = {'diff': point[n][k] - point['reader'][k], 'ci': np.nanpercentile(d, [lo, hi]).tolist(),
                           'p': two_sided_p(d[~np.isnan(d)])}
    return point, ci, per_reader, diffs


def anchor_report(results, n_joints):
    lines = ['# E4 inter-reader anchor (RheumaVIT)', '',
             'Each reader against the lower median of the other two readers; every model against that same '
             'reference on the same joints. 13 shared joints, 4 levels; mean over the 3 readers (reader 0 scored '
             'about half of the hands); 95% CI from a patient bootstrap (hier over runs). Difference = model - '
             'reader (negative = below inter-reader agreement).']
    for scope, (point, ci, per_reader, diffs) in results.items():
        lines += ['', f"## Scope: {scope} ({n_joints[scope]})", '',
                  '| Source | BACC4 [95% CI] | BACC4 - reader [95% CI] | QWK [95% CI] | QWK - reader [95% CI] | '
                  'QWK per reader 0 / 1 / 2 |', '|---|---|---|---|---|---|']
        for n in point:
            d = diffs.get(n)
            fd = lambda k: '' if d is None else f"{d[k]['diff']:+.3f} [{d[k]['ci'][0]:+.3f}, {d[k]['ci'][1]:+.3f}]"
            pr = ' / '.join(f"{per_reader[n][r]['qwk']:.2f}" for r in READERS)
            lines.append(f"| {n} | {point[n]['bacc4']:.3f} [{ci[n]['bacc4'][0]:.3f}, {ci[n]['bacc4'][1]:.3f}] | "
                         f"{fd('bacc4')} | {point[n]['qwk']:.3f} [{ci[n]['qwk'][0]:.3f}, {ci[n]['qwk'][1]:.3f}] | "
                         f"{fd('qwk')} | {pr} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('direct', 'fewshot', 'analyse', 'anchor'))
    p.add_argument('--manifest', default='runs/e4/rheumavit/manifest.csv')
    p.add_argument('--out', default='runs/e4')
    p.add_argument('--n-boot', type=int, default=10000)
    args = p.parse_args(argv)
    if args.step == 'direct':
        direct(args.manifest, args.out)
        return
    if args.step == 'fewshot':
        fewshot(args.manifest, args.out)
        return
    if args.step == 'anchor':
        results, n_joints = {}, {}
        for scope in ('test', 'all'):
            cells, patients = anchor_cells(args.out, args.manifest, scope)
            results[scope] = anchor(cells, patients, args.n_boot, levels=anchor_levels(cells))
            n_joints[scope] = f"{len(patients)} patients, reader-scored joints " + ' / '.join(
                str(len(cells[('reader', r)][0])) for r in READERS)
        text = anchor_report(results, n_joints)
        dest = Path(args.out) / 'analysis'
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'anchor.md').write_text(text)
        (dest / 'anchor.json').write_text(json.dumps({s: dict(zip(('point', 'ci', 'per_reader', 'diffs'), r))
                                                      for s, r in results.items()}, indent=2))
        print(text)
        return
    runs = load_predictions(args.out, args.manifest)
    levels = {c: run_levels([r.name for r in source_runs(c)]) for c in CONFIGS}
    point, ci, per_run, contrasts = analyse(runs, args.n_boot, levels=levels)
    ext = runs['K3'][0][1]
    text = report(point, ci, per_run, contrasts, {'patients': ext['patient'].nunique(), 'joints': len(ext)})
    dest = Path(args.out) / 'analysis'
    dest.mkdir(parents=True, exist_ok=True)
    cells = load_fewshot(args.out, args.manifest)
    if cells is not None:
        fs = fewshot_analysis(cells, args.n_boot)
        text += fewshot_report(*fs)
        (dest / 'fewshot.json').write_text(json.dumps(
            {'point': {f'{c}_k{k}': v for (c, k), v in fs[0].items()},
             'sd': {f'{c}_k{k}': v for (c, k), v in fs[1].items()},
             'ci': {f'{c}_k{k}': v for (c, k), v in fs[2].items()}, 'contrasts': fs[3]}, indent=2))
    (dest / 'report.md').write_text(text)
    (dest / 'direct.json').write_text(json.dumps({'point': point, 'ci': ci, 'per_run': per_run,
                                                  'contrasts': contrasts}, indent=2))
    print(text)


if __name__ == '__main__':
    main()
