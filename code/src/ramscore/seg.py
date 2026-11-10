"""Bone segmentation for the anatomy-guided E6 ablation (proposal 5.1).

A U-Net (ImageNet ResNet34 encoder) learns the union of the RAM-H1200 bone masks on the train hands, with the best
epoch chosen by validation Dice. Its predictions, not the reference masks, feed the classifier, so test joints get
what a deployed model would see. Excluded from the union: SoftTissue, Intravenous cannula, Metal Implant, Ring and
the parent category.

  train    -> <out>/seg/{model_best.pt,metrics.json}
  predict  -> <out>/masks/<split>/<case>/<joint>.png, 0/255, 224 x 224, cut with the annotated box of each joint

Train hands are predicted by a model that saw them, so their masks may be slightly better than test masks.

Usage: uv run python -m ramscore.seg --out runs/e6abl
"""
import argparse
import json
import math
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import BE_JOINTS

ROOT = os.environ.get('RAM_H1200_ROOT', '/data/Smart-X/tesis-fix/dataset/RAM-H1200-v1')
NOT_BONE = {'bone-6U7D-RuyD', 'SoftTissue', 'Intravenous cannula', 'Metal Implant', 'Ring'}
SIZE = 768


def rle_decode(counts, height, width):
    """COCO compressed RLE string -> height x width uint8 mask (the pycocotools LEB128 variant)."""
    data = counts.encode() if isinstance(counts, str) else counts
    runs, i = [], 0
    while i < len(data):
        value, shift, more = 0, 0, True
        while more:
            c = data[i] - 48
            value |= (c & 0x1f) << (5 * shift)
            more = bool(c & 0x20)
            i += 1
            shift += 1
            if not more and c & 0x10:
                value |= -1 << (5 * shift)
        if len(runs) > 2:
            value += runs[-2]
        runs.append(value)
    flat = np.zeros(height * width, dtype=np.uint8)
    pos, on = 0, 0
    for run in runs:
        flat[pos:pos + run] = on
        pos += run
        on = 1 - on
    return flat.reshape((height, width), order='F')


def bone_annotations(root, split):
    """{file name: (height, width, [RLE counts of the bone categories])}."""
    j = json.loads((Path(root) / 'Segmentation' / split / '_annotations_bone_rle.coco.json').read_text())
    keep = {c['id'] for c in j['categories'] if c['name'] not in NOT_BONE}
    out = {i['id']: (int(i['height']), int(i['width']), i['file_name'], []) for i in j['images']}
    for a in j['annotations']:
        if a['category_id'] in keep:
            out[a['image_id']][3].append(a['segmentation']['counts'])
    return {name: (h, w, rles) for h, w, name, rles in out.values()}


def to_square(img, size=SIZE, interpolation=cv2.INTER_LINEAR):
    """Pad at the bottom and right to a square, then resize; returns the image and the side before resizing."""
    h, w = img.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=img.dtype)
    padded[:h, :w] = img
    return cv2.resize(padded, (size, size), interpolation=interpolation), side


def from_square(prob, h, w, side):
    return cv2.resize(prob, (side, side), interpolation=cv2.INTER_LINEAR)[:h, :w]


class HandDataset(Dataset):
    def __init__(self, root, split, transform=None, limit=0):
        items = sorted(bone_annotations(root, split).items())
        items = items[:limit] if limit else items
        self.names, self.shapes, self.images, self.masks = [], [], [], []
        for name, (h, w, rles) in items:
            img = cv2.imread(str(Path(root) / 'Segmentation' / split / name), cv2.IMREAD_GRAYSCALE)
            mask = np.zeros((h, w), dtype=np.uint8)
            for counts in rles:
                mask |= rle_decode(counts, h, w)
            self.names.append(name)
            self.shapes.append((h, w, max(h, w)))
            self.images.append(to_square(img)[0])
            self.masks.append(to_square(mask, interpolation=cv2.INTER_NEAREST)[0])
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img, mask = self.images[i], self.masks[i]
        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out['image'], out['mask']
        return torch.from_numpy(img.astype(np.float32) / 255)[None], torch.from_numpy(mask.astype(np.float32))[None]


class UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model('resnet34', features_only=True, pretrained=pretrained, in_chans=1)
        ch = self.encoder.feature_info.channels()
        self.blocks = nn.ModuleList()
        up = ch[-1]
        for skip in reversed(ch[:-1]):
            self.blocks.append(nn.Sequential(nn.Conv2d(up + skip, skip, 3, padding=1), nn.BatchNorm2d(skip), nn.ReLU(True),
                                             nn.Conv2d(skip, skip, 3, padding=1), nn.BatchNorm2d(skip), nn.ReLU(True)))
            up = skip
        self.head = nn.Conv2d(up, 1, 1)

    def forward(self, x):
        features = self.encoder(x)
        z = features[-1]
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            z = block(torch.cat([F.interpolate(z, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
        return F.interpolate(self.head(z), size=x.shape[-2:], mode='bilinear', align_corners=False)


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    return 1 - ((2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def dice_score(pred, target):
    """Dice of two boolean arrays; 1 if both are empty."""
    total = pred.sum() + target.sum()
    return float(2 * (pred & target).sum() / total) if total else 1.0


@torch.no_grad()
def evaluate(net, ds, device, batch_size=4):
    net.eval()
    scores = []
    for x, y in DataLoader(ds, batch_size=batch_size, num_workers=2):
        with torch.autocast(device.type):
            p = torch.sigmoid(net(x.to(device)).float()).cpu().numpy() > 0.5
        scores += [dice_score(a, b) for a, b in zip(p, y.numpy() > 0.5)]
    return float(np.mean(scores))


def train(args, device):
    dest = Path(args.out) / 'seg'
    dest.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    aug = A.Compose([A.ShiftScaleRotate(rotate_limit=10), A.RandomBrightnessContrast(p=0.5)])
    train_ds = HandDataset(args.root, 'train', aug, args.limit)
    val_ds = HandDataset(args.root, 'val', limit=args.limit)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    net = UNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler()
    best = -math.inf
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type):
                z = net(x).float()
                loss = F.binary_cross_entropy_with_logits(z, y) + dice_loss(z, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        val = evaluate(net, val_ds, device)
        if val > best:
            best = val
            torch.save({'model': net.state_dict(), 'epoch': epoch, 'val_dice': val}, dest / 'model_best.pt')
        print(f'epoch {epoch:3d} loss {np.mean(losses):.4f} val dice {val:.4f}', flush=True)
    state = torch.load(dest / 'model_best.pt', map_location=device)
    net.load_state_dict(state['model'])
    test = evaluate(net, HandDataset(args.root, 'test', limit=args.limit), device)
    (dest / 'metrics.json').write_text(json.dumps({'epoch': state['epoch'], 'val_dice': state['val_dice'],
                                                   'test_dice': test}, indent=2))
    print(f"best epoch {state['epoch']}: val dice {state['val_dice']:.4f}, test dice {test:.4f}", flush=True)
    return net


def joint_boxes(root, split):
    """{file name: [(joint stem, x, y, w, h)]} for the 16 erosion joints."""
    j = json.loads((Path(root) / 'SvdH_Scoring' / 'SvdH_BE_Scoring' / split /
                    '_annotations_be_joint_detection.coco.json').read_text())
    names = {c['id']: c['name'] for c in j['categories']}
    files = {i['id']: i['file_name'] for i in j['images']}
    out = {}
    for a in j['annotations']:
        if names[a['category_id']] in BE_JOINTS:
            out.setdefault(files[a['image_id']], []).append((names[a['category_id']], *a['bbox']))
    return out


@torch.no_grad()
def predict_masks(net, args, device):
    net.eval()
    stats = {}
    for split in ('train', 'val', 'test'):
        ds = HandDataset(args.root, split, limit=args.limit)
        boxes = joint_boxes(args.root, split)
        fg, dice, n = [], [], 0
        for i, name in enumerate(ds.names):
            x = torch.from_numpy(ds.images[i].astype(np.float32) / 255)[None, None].to(device)
            with torch.autocast(device.type):
                prob = torch.sigmoid(net(x).float())[0, 0].cpu().numpy()
            h, w, side = ds.shapes[i]
            pred = from_square(prob, h, w, side) > 0.5
            ref = from_square(ds.masks[i].astype(np.float32), h, w, side) > 0.5    # reference masks are 0/1
            case = Path(name).stem
            dest = Path(args.out) / 'masks' / split / case
            dest.mkdir(parents=True, exist_ok=True)
            for joint, bx, by, bw, bh in boxes.get(name, []):
                x0, y0 = max(int(math.floor(bx)), 0), max(int(math.floor(by)), 0)
                x1, y1 = int(math.ceil(bx + bw)), int(math.ceil(by + bh))
                crop, target = pred[y0:y1, x0:x1], ref[y0:y1, x0:x1]
                fg.append(float(crop.mean()))
                dice.append(dice_score(crop, target))
                n += 1
                roi = cv2.resize(crop.astype(np.uint8) * 255, (224, 224), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(dest / f'{joint}.png'), roi)
        stats[split] = {'rois': n, 'mean_foreground': float(np.mean(fg)), 'roi_dice': float(np.mean(dice))}
        print(split, stats[split], flush=True)
    (Path(args.out) / 'seg' / 'roi_stats.json').write_text(json.dumps(stats, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=ROOT)
    p.add_argument('--out', default='runs/e6abl')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--limit', type=int, default=0, help='First N hands per split (smoke test)')
    args = p.parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict_masks(train(args, device), args, device)


if __name__ == '__main__':
    main()
