"""RQ1-07: intermediate supervised pre-training on RSNA bone age (M005), addendum 6.

  prepare   read the Kaggle zip (kmader/rsna-bone-age), pad every training hand radiograph to a square and resize it
            to 512 x 512 (grayscale PNG), write labels.csv (id, boneage, male) and SHA256SUMS; the zip can then go.
  pretrain  ImageNet ResNet34 (1 channel) -> regression of the z-scored bone age with MSE (M005): SGD lr 1e-3,
            momentum 0.9, wd 1e-3, 50 epochs, base augmentation, at 384 x 384; a 90/10 split of the 12,611 labelled
            images (seeded); the checkpoint with the lowest validation MAE becomes encoder.pt ({'model': state dict},
            the regression head is dropped by load_backbone at fine-tuning).

Usage: python -m ramscore.boneage prepare --zip <rsna-bone-age.zip> --out ../data/external/rsna-bone-age/png512
       python -m ramscore.boneage pretrain --data ../data/external/rsna-bone-age/png512 --name rq1-07_boneage_resnet34 --out ../runs/rq1_pretrain --amp
"""
import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from functools import partial
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PREPARED = 512


def pad_square(img):
    """Pad a grayscale image with zeros to a centred square (the hand keeps its aspect ratio)."""
    h, w = img.shape
    s = max(h, w)
    out = np.zeros((s, s), img.dtype)
    y, x = (s - h) // 2, (s - w) // 2
    out[y:y + h, x:x + w] = img
    return out


def prepare(zip_path, out, size=PREPARED):
    out = Path(out)
    (out / 'images').mkdir(parents=True, exist_ok=False)
    sums, kept = [], []
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        csv_name = next(n for n in names if n.endswith('boneage-training-dataset.csv'))
        labels = pd.read_csv(io.BytesIO(z.read(csv_name)))
        pngs = {Path(n).stem: n for n in names if '/boneage-training-dataset/' in n and n.endswith('.png')}
        for row in labels.itertuples():
            member = pngs.get(str(row.id))
            if member is None:
                continue
            img = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f'cannot decode {member}')
            img = cv2.resize(pad_square(img), (size, size), interpolation=cv2.INTER_AREA)
            buf = cv2.imencode('.png', img)[1].tobytes()
            (out / 'images' / f'{row.id}.png').write_bytes(buf)
            sums.append(f'{hashlib.sha256(buf).hexdigest()}  images/{row.id}.png')
            kept.append({'id': int(row.id), 'boneage': float(row.boneage), 'male': bool(row.male)})
    pd.DataFrame(kept).to_csv(out / 'labels.csv', index=False)
    (out / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    print(f'{len(kept)} images of {len(labels)} labels -> {out}')


def split_ids(ids, seed, val_share=0.1):
    """Seeded 90/10 split of the labelled images (the challenge validation set is not in the Kaggle copy)."""
    ids = np.array(sorted(ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_val = int(round(val_share * len(ids)))
    return ids[perm[n_val:]].tolist(), ids[perm[:n_val]].tolist()


class BoneAgeDataset(Dataset):
    def __init__(self, root, ids, targets, size, transform=None):
        self.root, self.ids, self.transform = Path(root), list(ids), transform
        self.targets = np.asarray(targets, dtype=np.float32)
        self.cache = np.stack([cv2.resize(cv2.imread(str(self.root / 'images' / f'{i}.png'), cv2.IMREAD_GRAYSCALE),
                                          (size, size), interpolation=cv2.INTER_AREA) for i in self.ids])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        img = self.cache[k]
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return {'img': torch.from_numpy(img.astype(np.float32) / 255.0)[None], 'age': torch.tensor(self.targets[k])}


def zscore(ages, mean, std):
    return (np.asarray(ages, dtype=np.float32) - mean) / std


@torch.no_grad()
def evaluate(net, loader, device, mean, std):
    """Validation MAE in months."""
    net.eval()
    err = []
    for b in loader:
        pred = net(b['img'].to(device)).float()[:, 0].cpu().numpy()
        err.append(np.abs(pred - b['age'].numpy()) * std)
    return float(np.concatenate(err).mean())


def pretrain(args):
    from .data import seed_augmentation_worker
    from .models import build_model
    from .train import seed_everything

    run_dir = Path(args.out) / args.name
    if (run_dir / 'done.json').exists():
        print(f'{run_dir} already finished')
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = pd.read_csv(Path(args.data) / 'labels.csv')
    if args.limit:
        labels = labels.iloc[:args.limit]
    age = dict(zip(labels['id'], labels['boneage']))
    train_ids, val_ids = split_ids(labels['id'], args.seed)
    mean, std = float(np.mean([age[i] for i in train_ids])), float(np.std([age[i] for i in train_ids]))
    aug = A.Compose([A.ShiftScaleRotate(rotate_limit=10), A.RandomBrightnessContrast(p=0.5)], seed=args.seed)
    t0 = time.time()
    ds = {'train': BoneAgeDataset(args.data, train_ids, zscore([age[i] for i in train_ids], mean, std), args.size, aug),
          'val': BoneAgeDataset(args.data, val_ids, zscore([age[i] for i in val_ids], mean, std), args.size)}
    print(f"Bone age: train {len(ds['train'])}, val {len(ds['val'])} images at {args.size} px "
          f"({time.time() - t0:.0f}s); mean {mean:.1f}, sd {std:.1f} months")
    loaders = {'train': DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True, drop_last=True,
                                   num_workers=args.workers, pin_memory=True,
                                   generator=torch.Generator().manual_seed(args.seed),
                                   worker_init_fn=partial(seed_augmentation_worker, args.seed) if args.workers else None),
               'val': DataLoader(ds['val'], batch_size=64, num_workers=args.workers)}
    (run_dir / 'args.json').write_text(json.dumps({**vars(args), 'age_mean': mean, 'age_sd': std,
                                                   'n_train': len(train_ids), 'n_val': len(val_ids)}, indent=2))
    net = build_model('resnet34', 1, 'softmax', pretrained=not args.no_pretrained).to(device)   # one regression output
    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    start, best, history = 0, math.inf, []
    ckpt = run_dir / 'checkpoint.pt'
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        sched.load_state_dict(state['sched'])
        scaler.load_state_dict(state['scaler'])
        start, best, history = state['epoch'] + 1, state['best'], state['history']
        print(f'Resumed at epoch {start}')
    for epoch in range(start, args.epochs):
        net.train()
        losses, t0 = [], time.time()
        for b in loaders['train']:
            img, target = b['img'].to(device, non_blocking=True), b['age'].to(device)
            with torch.autocast(device.type, enabled=args.amp):
                loss = F.mse_loss(net(img).float()[:, 0], target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        mae = evaluate(net, loaders['val'], device, mean, std)
        improved = mae < best
        if improved:
            best = mae
            torch.save({'model': net.state_dict(), 'epoch': epoch, 'val_mae_months': mae}, run_dir / 'encoder.pt')
        history.append({'epoch': epoch, 'train_mse': float(np.mean(losses)), 'val_mae_months': mae,
                        'seconds': time.time() - t0})
        print(f"epoch {epoch:3d} mse {history[-1]['train_mse']:.4f} val MAE {mae:.2f} months "
              f"{history[-1]['seconds']:.0f}s{' *' if improved else ''}")
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
        torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'history': history}, ckpt)
    (run_dir / 'done.json').write_text(json.dumps({'epochs': args.epochs, 'best_val_mae_months': best}))


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['prepare', 'pretrain'])
    p.add_argument('--zip')
    p.add_argument('--data', help='prepared folder (labels.csv, images/)')
    p.add_argument('--out', default='runs')
    p.add_argument('--name', default='rq1-07_boneage_resnet34')
    p.add_argument('--size', type=int, default=384)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='first N labelled images (smoke test)')
    p.add_argument('--no-pretrained', action='store_true', help='Tests only')
    args = p.parse_args(argv)
    if args.command == 'prepare' and not args.zip:
        p.error('prepare needs --zip and --out')
    if args.command == 'pretrain' and not args.data:
        p.error('pretrain needs --data')
    return args


def main(argv=None):
    args = get_args(argv)
    if args.command == 'prepare':
        prepare(args.zip, args.out)
    else:
        pretrain(args)


if __name__ == '__main__':
    main()
