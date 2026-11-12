"""Self-supervised pre-training on the unlabelled joint-ROI pool (E1: proposal A3-A5).

  A3  --method mae  --model vit_small_patch16_224
  A4  --method dino --model resnet34 | vit_small_patch16_224
  A5  A3 or A4 + --chrono-weight 1 (ChronoCon on the longitudinal series)
  RQ1-08 --method simclr --model resnet34;  RQ1-10 --method byol --model resnet34 (addendum 4: the E1 DINO ResNet34
         recipe, with its two asymmetric global crops and no local crops, so that only the objective differs)

Writes runs/<name>/: args.json, history.csv, checkpoint.pt (full state, for
--resume) and encoder.pt ({'model': backbone state dict}; the DINO teacher), plus
encoder_ep<N>.pt every --save-every epochs. Fine-tune with
  python -m ramscore.train --model <same backbone> --init-from runs/<name>/encoder.pt

Usage: python -m ramscore.pretrain --name e1_dino_resnet34 --method dino --model resnet34 --amp
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pool import W600_ROOT, GroupBatchSampler, PoolDataset, chrono_groups, load_pool
from .ssl import BYOL, DINO, MAE, Crops, MultiCrop, SimCLR, chronocon_loss, cosine_schedule
from .train import seed_everything

# Per-method defaults (MAE: He et al., 2022; DINO: Caron et al., 2021, scaled to one 12 GB GPU)
DEFAULTS = {
    'mae': dict(epochs=400, batch_size=128, lr=1.5e-4, weight_decay=0.05, weight_decay_end=0.05, warmup_epochs=20),
    'dino': dict(epochs=100, batch_size=64, lr=5e-4, weight_decay=0.04, weight_decay_end=0.4, warmup_epochs=10),
    # RQ1-08/10 (addendum 4): the DINO optimiser recipe; batch 128 x 2 views costs about one DINO step (2 x 224 + 6 x 96)
    'simclr': dict(epochs=100, batch_size=128, lr=5e-4, weight_decay=0.04, weight_decay_end=0.4, warmup_epochs=10),
    'byol': dict(epochs=100, batch_size=128, lr=5e-4, weight_decay=0.04, weight_decay_end=0.4, warmup_epochs=10),
}
EMA = ('dino', 'byol')     # methods with a momentum teacher / target


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True)
    p.add_argument('--out', default='runs')
    p.add_argument('--method', required=True, choices=list(DEFAULTS))
    p.add_argument('--model', default='resnet34', help='timm backbone name (MAE needs a ViT)')
    p.add_argument('--pool', default='h1200', choices=['h1200', 'h1200+w600'])
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--lr', type=float, help='Base learning rate per 256 images')
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--weight-decay-end', type=float)
    p.add_argument('--warmup-epochs', type=int)
    p.add_argument('--clip-grad', type=float, default=3.0)
    p.add_argument('--mask-ratio', type=float, default=0.75, help='MAE')
    p.add_argument('--out-dim', type=int, default=4096, help='DINO prototypes')
    p.add_argument('--n-local', type=int, default=6, help='DINO local crops')
    p.add_argument('--teacher-temp', type=float, default=0.04, help='DINO')
    p.add_argument('--momentum', type=float, default=0.996, help='DINO teacher EMA, cosine to 1')
    p.add_argument('--freeze-last-epochs', type=int, default=1, help='DINO: keep the prototype layer fixed')
    p.add_argument('--temperature', type=float, default=0.1, help='SimCLR NT-Xent temperature')
    p.add_argument('--chrono-weight', type=float, default=0.0, help='ChronoCon weight; 0 = off (A3/A4), >0 = A5')
    p.add_argument('--chrono-groups', type=int, default=8, help='Series per ChronoCon batch')
    p.add_argument('--chrono-max-visits', type=int, default=4)
    p.add_argument('--chrono-tau', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--resume', action='store_true', help='Continue from <out>/<name>/checkpoint.pt')
    p.add_argument('--limit', type=int, default=0, help='Use only the first patients, about N pool images (smoke test)')
    args = p.parse_args(argv)
    if args.chrono_weight and args.method not in ('mae', 'dino'):
        p.error('--chrono-weight is defined for mae and dino only')
    for k, v in DEFAULTS[args.method].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def param_groups(params):
    """No weight decay on biases, norm weights and tokens (1-d or scalar tensors)."""
    params = [p for p in params if p.requires_grad]
    return [{'params': [p for p in params if p.ndim > 1], 'decay': True},
            {'params': [p for p in params if p.ndim <= 1], 'decay': False}]


def forever(loader):
    while True:
        yield from loader


def main(argv=None):
    args = get_args(argv)
    run_dir = Path(args.out) / args.name
    if run_dir.exists() and not args.resume:
        raise SystemExit(f'{run_dir} exists; pass --resume to continue it')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True   # speed over bit-exactness
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    records = load_pool(w600_root=W600_ROOT if args.pool == 'h1200+w600' else None)
    if args.limit:
        # whole patients, so that their chronological series stay intact
        by_patient = {}
        for r in records:
            by_patient.setdefault(r['patient'], []).append(r)
        records = []
        for patient in sorted(by_patient):
            if len(records) >= args.limit:
                break
            records += by_patient[patient]
    t0 = time.time()
    views = {'dino': MultiCrop(args.n_local), 'mae': Crops(1)}.get(args.method, MultiCrop(0))   # SimCLR/BYOL: 2 global
    ds = PoolDataset(records, views)
    print(f'Pool: {len(ds)} ROIs from {len({r["patient"] for r in records})} patients, loaded in {time.time() - t0:.0f}s')
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0, generator=g)

    chrono = None
    if args.chrono_weight > 0:
        groups = chrono_groups(records)
        cds = copy.copy(ds)                                      # shares the image cache
        cds.views = Crops(2)                                     # double-crop (Zha et al., 2023)
        cds.group = np.full(len(records), -1)
        cds.set_groups(groups)
        sampler = GroupBatchSampler(groups, args.chrono_groups, args.chrono_max_visits, seed=args.seed)
        if len(sampler) == 0:
            raise SystemExit(f'ChronoCon needs >= {args.chrono_groups} series, the pool has {len(groups)}')
        chrono = forever(DataLoader(cds, batch_sampler=sampler, num_workers=min(args.workers, 2), pin_memory=True))
        print(f'ChronoCon: {len(groups)} series from {len({records[i[0]]["patient"] for i in groups})} patients')

    model = {'mae': lambda: MAE(args.model, args.mask_ratio), 'dino': lambda: DINO(args.model, args.out_dim),
             'simclr': lambda: SimCLR(args.model, temperature=args.temperature),
             'byol': lambda: BYOL(args.model)}[args.method]().to(device)
    student = model.student_parameters() if args.method in EMA else list(model.parameters())
    opt = torch.optim.AdamW(param_groups(student))
    scaler = torch.amp.GradScaler(enabled=args.amp)
    steps = len(loader)
    total = args.epochs * steps
    lr_at = cosine_schedule(args.lr * args.batch_size / 256, args.min_lr, total, args.warmup_epochs * steps)
    wd_at = cosine_schedule(args.weight_decay, args.weight_decay_end, total)
    momentum_at = cosine_schedule(args.momentum, 1.0, total)

    start, history = 0, []
    ckpt_path = run_dir / 'checkpoint.pt'
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        scaler.load_state_dict(state['scaler'])
        start, history = state['epoch'] + 1, state['history']
        print(f'Resumed at epoch {start}')

    encoder = model.teacher_backbone if args.method == 'dino' else model.backbone
    for epoch in range(start, args.epochs):
        model.train()
        t0, ssl_losses, chrono_losses = time.time(), [], []
        for it, batch in enumerate(tqdm(loader, desc=f'epoch {epoch}', leave=False)):
            step = epoch * steps + it
            for group in opt.param_groups:
                group['lr'] = lr_at[step]
                group['weight_decay'] = wd_at[step] if group['decay'] else 0.0
            views = [v.to(device, non_blocking=True) for v in batch['views']]
            with torch.autocast(device.type, enabled=args.amp):
                loss = (model(views[0]) if args.method == 'mae' else model(views, args.teacher_temp)
                        if args.method == 'dino' else model(views))
                ssl_losses.append(loss.item())
                if chrono is not None:
                    cb = next(chrono)
                    feats = model.backbone(torch.cat(cb['views']).to(device, non_blocking=True))
                    n_views = len(cb['views'])
                    lc = chronocon_loss(feats, cb['group'].repeat(n_views).to(device),
                                        cb['time'].repeat(n_views).to(device), args.chrono_tau)
                    chrono_losses.append(lc.item())
                    loss = loss + args.chrono_weight * lc
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student, args.clip_grad)
            if args.method == 'dino' and epoch < args.freeze_last_epochs:
                model.head.last.grad = None
            scaler.step(opt)
            scaler.update()
            if args.method in EMA:
                model.update_teacher(momentum_at[step])
        history.append({'epoch': epoch, 'ssl_loss': float(np.mean(ssl_losses)),
                        'chrono_loss': float(np.mean(chrono_losses)) if chrono_losses else None,
                        'lr': float(lr_at[step]), 'seconds': time.time() - t0,
                        'gpu_peak_gb': torch.cuda.max_memory_reserved() / 1e9 if device.type == 'cuda' else None})
        h = history[-1]
        print(f"epoch {epoch:3d} ssl {h['ssl_loss']:.4f}"
              + (f" chrono {h['chrono_loss']:.4f}" if chrono_losses else '') + f" lr {h['lr']:.2e} {h['seconds']:.0f}s"
              + (f" gpu {h['gpu_peak_gb']:.1f}GB" if h['gpu_peak_gb'] else ''))
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
        enc_state = {'model': encoder.state_dict(), 'epoch': epoch, 'method': args.method, 'backbone': args.model}
        torch.save(enc_state, run_dir / 'encoder.pt')
        if (epoch + 1) % args.save_every == 0:
            torch.save(enc_state, run_dir / f'encoder_ep{epoch + 1}.pt')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': epoch, 'history': history}, ckpt_path)
    (run_dir / 'done.json').write_text(json.dumps({'epochs': args.epochs}))   # queue marker: encoder.pt is final


if __name__ == '__main__':
    main()
