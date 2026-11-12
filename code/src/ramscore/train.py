"""Train and test one joint-level BE scoring model on RAM-H1200.

Defaults reproduce the official benchmark (train_score_cls.sh / main_score_cls.py):
ResNet34 from scratch, 1-channel 224 x 224 ROIs, ordinal BCE head, AdamW
(lr 1e-4, wd 1e-3), cosine schedule to 1e-6 over 200 epochs, batch 32, best
checkpoint by validation QWK, no oversampling and no class weights.

Factor B options (see models.py): --head, --class-weight, --oversample,
--logit-adjust. Decoupled retraining (B6): --init-from <run>/model_best.pt
--freeze-backbone --oversample 1, which keeps the encoder and retrains a fresh classifier.
Grade-0 undersampling (E7): --undersample-zero, on the train split only; validation and test stay whole.
SupCon + CE (RQ1-12) and its grade-distance weighted negatives (RQ1-13, after M219): --supcon-weight, --supcon-alpha.
Visit-pair penalty (E9): --pair-weight, on consecutive visits of one joint from every training patient, labelled or
not; each step adds --pair-batch pairs to the labelled batch, and inference still takes a single image.

Usage: python -m ramscore.train --name e0_resnet34_s2026 [--seed 2026] [...]
"""
import argparse
import json
import math
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timm.optim import param_groups_layer_decay
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import (AUGMENTATIONS, NATIVE_SCORES, BEDataset, RotatingZeroSampler, VisitPairDataset, attach_masks,
                   get_transform, load_records, seed_augmentation_worker, subsample_patients, undersample_summary,
                   undersample_zero)
from .metrics import bootstrap_ci, compute_metrics
from .models import (HEADS, Criterion, ProjectionHead, add_input_channels, build_model, decode, expected_grade,
                     freeze_backbone, load_backbone, load_full, monotonic_penalty, pooled_and_logits, supcon_loss)

DEFAULT_ROOT = '/data/Smart-X/tesis-fix/dataset/RAM-H1200-v1/SvdH_Scoring/SvdH_BE_Scoring'


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True, help='Run name; outputs go to <out>/<name>/')
    p.add_argument('--data-root', default=DEFAULT_ROOT)
    p.add_argument('--out', default='runs')
    p.add_argument('--model', default='ResNet34', help='Benchmark alias or any timm model name')
    p.add_argument('--pretrained', action='store_true', help='ImageNet weights (benchmark: off)')
    p.add_argument('--head', default='ordinal', choices=HEADS)
    p.add_argument('--class-weight', action='store_true')
    p.add_argument('--oversample', type=float, default=0.0,
                   help='Sampling weight exponent on 1/class count; 0 = off (benchmark: off)')
    p.add_argument('--logit-adjust', type=float, default=0.0,
                   help='Logit adjustment tau on the training loss; 1 = balanced softmax; 0 = off')
    p.add_argument('--undersample-zero', default='off', choices=['off', 'permanent', 'rotate'],
                   help='Grade 0 to 1:1 against all other grades, same share per joint type: dropped for good, or '
                        'a new subset every epoch (E7); off = benchmark')
    p.add_argument('--aug', default='base', choices=AUGMENTATIONS,
                   help='Training augmentation: base = benchmark recipe; none, geom, intensity = its ablations (E8)')
    p.add_argument('--seed-aug', action='store_true',
                   help='Tie the augmentation to --seed; off = unseeded, as in every run before E8')
    p.add_argument('--pair-weight', type=float, default=0.0,
                   help='Weight of the penalty on a fall of the expected grade between visits (E9); 0 = off')
    p.add_argument('--pair-margin', type=float, default=0.25,
                   help='Fall of the expected grade that goes unpenalised, in grade positions')
    p.add_argument('--pair-batch', type=int, default=16, help='Visit pairs per step, on top of the labelled batch')
    p.add_argument('--supcon-weight', type=float, default=0.0,
                   help='Weight of the supervised contrastive loss added to the classification loss (RQ1-12/13); '
                        'two augmented views per joint, the classification loss sees the first only; 0 = off')
    p.add_argument('--supcon-temp', type=float, default=0.1, help='SupCon temperature (Khosla et al., 2020)')
    p.add_argument('--supcon-alpha', type=float, default=None,
                   help='Grade-distance weight of SupCon negatives (M219 eq. 6, alpha); unset = plain SupCon')
    p.add_argument('--init-from', help='Checkpoint whose encoder initialises the model (classifier is reset)')
    p.add_argument('--freeze-backbone', action='store_true',
                   help='Train only the classifier (linear probe); needs --init-from or --pretrained')
    p.add_argument('--keep-head', action='store_true',
                   help='With --init-from: load the whole checkpoint, classifier included (LP-FT stage 2, RQ1-17)')
    p.add_argument('--label-fraction', type=float, default=1.0, help='Patient-level fraction of training labels')
    p.add_argument('--fraction-seed', type=int, default=0)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--min-steps', type=int, default=0,
                   help='Raise --epochs until training takes at least this many optimizer steps (small fractions)')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--layer-decay', type=float, default=0.0,
                   help='Layer-wise LR decay (MAE fine-tuning recipe, ViT; no weight decay on norms and biases); 0 = off')
    p.add_argument('--warmup', type=float, default=0.0,
                   help='Linear warm-up over this fraction of the epochs before the cosine schedule; 0 = off')
    p.add_argument('--mask-root', help='Add a bone-mask input channel (E6 ablation; masks written by ramscore.seg)')
    p.add_argument('--levels', type=int, default=5, choices=[5, 3],
                   help='Grades as classes: 5 = native, 3 = {0, 1, >= 2} (E6 ablation); patient draws stay the same')
    p.add_argument('--monitor', default='qwk', choices=['qwk', 'bacc4', 'bacc5', 'macro_f1'])
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--amp', action='store_true', help='Mixed precision (benchmark: off)')
    p.add_argument('--grad-checkpointing', action='store_true',
                   help='Recompute activations in the backward pass (memory for time; same result; ViT-L, RQ1-18)')
    p.add_argument('--n-boot', type=int, default=2000, help='Patient-level bootstrap resamples for test CIs')
    p.add_argument('--limit', type=int, default=0, help='Use only the first N records per split (smoke test)')
    return p.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(ds, args, train, generator):
    sampler = None
    if train and args.undersample_zero == 'rotate':
        sampler = RotatingZeroSampler(ds.records, args.seed)
    elif train and args.oversample > 0:
        counts = np.bincount(ds.labels, minlength=ds.num_classes)
        weights = (1.0 / np.maximum(counts, 1)) ** args.oversample
        sampler = WeightedRandomSampler(torch.as_tensor(weights[ds.labels], dtype=torch.double),
                                        num_samples=len(ds), replacement=True, generator=generator)
    worker_init = partial(seed_augmentation_worker, args.seed) if train and args.seed_aug else None
    return DataLoader(ds, batch_size=args.batch_size, shuffle=train and sampler is None, sampler=sampler,
                      drop_last=train, num_workers=args.workers, pin_memory=True,
                      persistent_workers=args.workers > 0, generator=generator, worker_init_fn=worker_init)


def check_init_args(args):
    """Linear probe needs pre-trained weights to freeze; LP-FT stage 2 (--keep-head) fine-tunes a whole checkpoint."""
    if args.freeze_backbone and not (args.init_from or args.pretrained):
        raise SystemExit('--freeze-backbone needs --init-from or --pretrained')
    if args.keep_head and (not args.init_from or args.freeze_backbone or args.mask_root):
        raise SystemExit('--keep-head needs --init-from and cannot be combined with --freeze-backbone or --mask-root')


def check_pair_args(args):
    """The penalty is defined on the softmax head and tested only with the plain E9 recipe."""
    if args.pair_weight and (args.head != 'softmax' or args.levels != 5 or args.mask_root or args.freeze_backbone
                             or args.undersample_zero != 'off' or args.class_weight or args.oversample):
        raise SystemExit('--pair-weight needs --head softmax and cannot be combined with --levels 3, --mask-root, '
                         '--freeze-backbone, --undersample-zero, --class-weight or --oversample')


def check_supcon_args(args):
    """SupCon is added to the plain fine-tuning recipe only; alpha is a weight in (0, 1]."""
    if args.supcon_alpha is not None and not args.supcon_weight:
        raise SystemExit('--supcon-alpha needs --supcon-weight')
    if args.supcon_alpha is not None and not 0 < args.supcon_alpha <= 1:
        raise SystemExit('--supcon-alpha must be in (0, 1]')
    if args.supcon_weight and (args.pair_weight or args.mask_root or args.freeze_backbone or args.keep_head
                               or args.undersample_zero != 'off' or args.layer_decay):
        raise SystemExit('--supcon-weight cannot be combined with --pair-weight, --mask-root, --freeze-backbone, '
                         '--keep-head, --undersample-zero or --layer-decay')


def supcon_step(net, proj, batch, target, criterion, args, num_classes, device):
    """One forward pass over both views, so BatchNorm sees them together. Classification loss on the first view only,
    as in the reference recipe; SupCon on both. Returns the classification loss and the SupCon loss."""
    img2 = batch['img2'].to(device, non_blocking=True)
    n = target.shape[0]
    feats, logits = pooled_and_logits(net, torch.cat([batch['img'].to(device, non_blocking=True), img2]))
    z = proj(feats.float())
    return (criterion(logits[:n].float(), target),
            supcon_loss(z, torch.cat([target, target]), args.supcon_temp, args.supcon_alpha, num_classes))


def make_pair_loader(ds, args):
    # own generator and worker streams, so the labelled loader keeps its order and its augmentation
    worker_init = partial(seed_augmentation_worker, args.seed + 1000) if args.seed_aug else None
    return DataLoader(ds, batch_size=args.pair_batch, shuffle=True, drop_last=True, num_workers=args.workers,
                      pin_memory=True, persistent_workers=args.workers > 0,
                      generator=torch.Generator().manual_seed(args.seed + 1), worker_init_fn=worker_init)


def endless(loader):
    while True:
        yield from loader


def paired_loss(net, img, target, pair, criterion, margin):
    """One forward pass over labelled joints and visit pairs, so BatchNorm sees them together.
    Returns the supervised loss, the penalty and the share of pairs whose fall exceeds the margin."""
    n = img.shape[0]
    logits = net(torch.cat([img, pair['early'], pair['late']])).float()
    early, late = logits[n:].chunk(2)
    violated = (expected_grade(early) - expected_grade(late) > margin).float().mean().item()
    return criterion(logits[:n], target), monotonic_penalty(early, late, margin), violated


@torch.no_grad()
def predict(net, loader, head, device, score_values):
    net.eval()
    rows = []
    for batch in loader:
        logits = net(batch['img'].to(device, non_blocking=True))
        pred = decode(logits.float(), head).cpu().numpy()
        for i, cls in enumerate(pred):
            rows.append({'case': batch['case'][i], 'patient': batch['patient'][i], 'joint': batch['joint'][i],
                         'gt': int(batch['score'][i]), 'pred': int(score_values[cls])})
    return pd.DataFrame(rows)


def main(argv=None):
    args = get_args(argv)
    check_init_args(args)
    if args.undersample_zero != 'off' and (args.class_weight or args.oversample or args.logit_adjust
                                           or args.freeze_backbone):
        raise SystemExit('--undersample-zero cannot be combined with --class-weight, --oversample, --logit-adjust '
                         'or --freeze-backbone')
    check_pair_args(args)
    check_supcon_args(args)
    run_dir = Path(args.out) / args.name
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits = {s: load_records(args.data_root, s) for s in ('train', 'val', 'test')}
    pair_pool = splits['train']     # every training patient; visit pairs read no label
    splits['train'] = subsample_patients(splits['train'], args.label_fraction, args.fraction_seed)
    score_values = NATIVE_SCORES
    if args.levels == 3:
        score_values = (0, 1, 2)
        splits = {s: [dict(x, score=min(x['score'], 2)) for x in r] for s, r in splits.items()}
    if args.mask_root:
        splits = {s: attach_masks(r, args.mask_root, s) for s, r in splits.items()}
    if args.limit:
        splits = {s: r[:args.limit] for s, r in splits.items()}
    if args.undersample_zero != 'off':
        summary = undersample_summary(splits['train'], args.undersample_zero, args.seed)
        (run_dir / 'undersample.json').write_text(json.dumps(summary, indent=2))
        if args.undersample_zero == 'permanent':
            splits['train'] = undersample_zero(splits['train'], args.seed)
    t0 = time.time()
    aug_seed = args.seed if args.seed_aug else None
    ds = {s: BEDataset(r, score_values, args.image_size, get_transform(s, args.aug, aug_seed)) for s, r in splits.items()}
    print(f'Loaded ROIs in {time.time() - t0:.0f}s: ' + ', '.join(
        f"{s} {len(d)} joints / {len({r['patient'] for r in d.records})} patients" for s, d in ds.items()))

    if args.supcon_weight:
        ds['train'].views = 2
    g = torch.Generator().manual_seed(args.seed)
    loaders = {s: make_loader(d, args, s == 'train', g) for s, d in ds.items()}
    args.epochs = max(args.epochs, math.ceil(args.min_steps / max(len(loaders['train']), 1)))
    (run_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))
    num_classes = len(score_values)
    net = build_model(args.model, num_classes, args.head, args.pretrained)
    if args.init_from and args.keep_head:
        load_full(net, args.init_from)
        print(f'Whole model, classifier included, from {args.init_from}')
    elif args.init_from:
        print(f'Encoder from {args.init_from}; fresh classifier: {load_backbone(net, args.init_from)}')
    if args.mask_root:
        add_input_channels(net)
    if args.freeze_backbone:
        freeze_backbone(net)
    if args.grad_checkpointing:
        net.set_grad_checkpointing(True)
    net.to(device)
    criterion = Criterion(args.head, ds['train'].labels, num_classes, args.class_weight, args.logit_adjust).to(device)
    proj = ProjectionHead(net.num_features).to(device) if args.supcon_weight else None
    extra = list(proj.parameters()) if proj is not None else []   # projection: trained, never saved
    if args.layer_decay:
        groups = param_groups_layer_decay(net, args.weight_decay, layer_decay=args.layer_decay)
        for g in groups:
            g['lr'] = args.lr * g.pop('lr_scale')
        opt = torch.optim.AdamW(groups, lr=args.lr)
    else:
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad] + extra, lr=args.lr,
                                weight_decay=args.weight_decay)
    if args.layer_decay or args.warmup:
        # same cosine to 1% as below, but as a factor so every layer group keeps its ratio
        warm = round(args.warmup * args.epochs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm else
                                                  0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (e - warm) /
                                                                                     (args.epochs - warm))))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    pairs = None
    if args.pair_weight:
        pool = VisitPairDataset(pair_pool, args.image_size, get_transform('train', args.aug, aug_seed))
        print(f'Visit pairs: {len(pool)} from {len({r["patient"] for r in pair_pool})} patients, '
              f'{args.pair_batch} per step, weight {args.pair_weight}, margin {args.pair_margin}')
        pairs = endless(make_pair_loader(pool, args))

    best, history = -np.inf, []
    for epoch in range(args.epochs):
        net.train()
        if proj is not None:
            proj.train()
        if args.freeze_backbone:
            net.eval()                      # keep BatchNorm statistics of the frozen encoder
            net.get_classifier().train()
        losses, penalties, violations, contrastive = [], [], [], []
        for batch in tqdm(loaders['train'], desc=f'epoch {epoch}', leave=False):
            img, target = batch['img'].to(device, non_blocking=True), batch['label'].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=args.amp):
                if proj is not None:
                    supervised, con = supcon_step(net, proj, batch, target, criterion, args, num_classes, device)
                    loss = supervised + args.supcon_weight * con
                    contrastive.append(con.item())
                elif pairs is None:
                    supervised = criterion(net(img).float(), target)
                    loss = supervised
                else:
                    pair = {k: v.to(device, non_blocking=True) for k, v in next(pairs).items()}
                    supervised, penalty, violated = paired_loss(net, img, target, pair, criterion, args.pair_margin)
                    loss = supervised + args.pair_weight * penalty
                    penalties.append(penalty.item())
                    violations.append(violated)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(supervised.item())
        sched.step()
        val_df = predict(net, loaders['val'], args.head, device, score_values)
        val = compute_metrics(val_df['gt'], val_df['pred'], score_values)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                        **{f'val_{k}': val[k] for k in ('qwk', 'bacc4', 'bacc5', 'macro_f1', 'mae')},
                        **({'pair_penalty': float(np.mean(penalties)), 'pair_violation': float(np.mean(violations))}
                           if pairs is not None else {}),
                        **({'supcon_loss': float(np.mean(contrastive))} if proj is not None else {})})
        improved = val[args.monitor] > best
        if improved:
            best = val[args.monitor]
            torch.save({'model': net.state_dict(), 'epoch': epoch, 'val': val}, run_dir / 'model_best.pt')
        print(f"epoch {epoch:3d} loss {history[-1]['train_loss']:.4f} val qwk {val['qwk']:.4f} "
              f"bacc5 {val['bacc5']:.4f} bacc4 {val['bacc4']:.4f}{' *' if improved else ''}")
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)

    state = torch.load(run_dir / 'model_best.pt', map_location=device)
    net.load_state_dict(state['model'])
    test_df = predict(net, loaders['test'], args.head, device, score_values)
    test_df.to_csv(run_dir / 'test_predictions.csv', index=False)
    result = {
        'best_epoch': state['epoch'], 'val': state['val'],
        'test': compute_metrics(test_df['gt'], test_df['pred'], score_values),
        'test_ci95_patient_bootstrap': bootstrap_ci(test_df, n_boot=args.n_boot, seed=args.seed) if args.n_boot else None,
    }
    (run_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    t = result['test']
    print(f"TEST bacc5 {t['bacc5']:.4f} bacc4 {t['bacc4']:.4f} qwk {t['qwk']:.4f} "
          f"macro_f1 {t['macro_f1']:.4f} acc {t['accuracy']:.4f} within1 {t['within1']:.4f}")


if __name__ == '__main__':
    main()
