"""RheumaVIT / AIRI hand radiographs (Stolpovsky et al., 2023) as joint ROIs in the RAM-H1200 format (E4).

Source layout: jpeg/<patient_id>.jpeg (one image, both hands), bboxes.csv (axis-aligned joint boxes),
scores.csv (erosion and JSN per joint for each of 3 readers), train_val_test_split.json (patients).

Geometry, checked on the images: the boxes of hand 'left' are in the right half of the image (x from w // 2),
those of hand 'right' in the left half mirrored, so every hand has its thumb on the left, as in RAM-H1200.
RAM-H1200 ROIs are axis-aligned squares (170 px for finger joints; the 'angle' in its COCO file is not
applied). Each hand is rescaled so the middle proximal phalanx (MCP3 -> PIP3 box centres) is REF_PHALANX px,
the median of RAM-H1200 train, and cut with the RAM-H1200 median box size of the joint.

Labels: erosion only, the 13 joints that map one-to-one onto RAM-H1200 (DIP1 = thumb IP, RC = distal
radius). The lumped 'wrist' box (RAM-H1200 scores lunate, scaphoid and trapezium apart) and DIP2-5 are left
out. Reference grade = lower median of the readers who scored the joint (median_low: with two readers who
disagree, the lower grade).

E4 split: the official train patients are the few-shot pool, official val + test the external test set.

Usage: python -m ramscore.rheumavit --src /home/sandimulyadi/Projects/Dataset/AIRINIIReumHands \
    --out runs/e4/rheumavit
"""
import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REF_PHALANX = 246.0
ROI_SIZE = {'R': 202, 'U': 113, 'CMC-T': 117}          # RAM-H1200 train medians; all others 170
FINGER_ROI = 170
# (RheumaVIT joint, finger) -> RAM-H1200 ROI stem
JOINTS = {
    ('MCP', 1): 'MCP-T', ('MCP', 2): 'MCP-I', ('MCP', 3): 'MCP-M', ('MCP', 4): 'MCP-R', ('MCP', 5): 'MCP-S',
    ('PIP', 2): 'PIP-I', ('PIP', 3): 'PIP-M', ('PIP', 4): 'PIP-R', ('PIP', 5): 'PIP-S',
    ('DIP', 1): 'IP', ('CMC', 1): 'CMC-T', ('ulna', 0): 'U', ('RC', 0): 'R',
}
READERS = (0, 1, 2)


def joint_key(df):
    return list(zip(df['joint'], df['finger'].fillna(0).astype(int)))


def hand_image(img, hand):
    half = img.shape[1] // 2
    return img[:, half:] if hand == 'left' else np.ascontiguousarray(img[:, :half][:, ::-1])


def crop_square(img, cx, cy, side):
    """Square of `side` px centred at (cx, cy); outside the image is black."""
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = max(0, -x0, -y0, x0 + side - img.shape[1], y0 + side - img.shape[0])
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = x0 + pad, y0 + pad
    return img[y0:y0 + side, x0:x0 + side]


def reference_scores(scores):
    """One row per (patient, hand, joint stem): per-reader erosion grades and their lower median."""
    e = scores[scores['disease'] == 'erosion'].dropna(subset=['score']).copy()
    e['stem'] = [JOINTS.get(k) for k in joint_key(e)]
    e = e.dropna(subset=['stem'])
    e['expert'] = e['expert'].astype(int)
    w = e.pivot_table(index=['patient_id', 'hand', 'stem'], columns='expert', values='score', aggfunc='first')
    w = w.reindex(columns=list(READERS))
    out = w.rename(columns={r: f'r{r}' for r in READERS}).reset_index()
    out['n_readers'] = w.notna().sum(axis=1).to_numpy()
    out['score'] = [int(statistics.median_low(v[~np.isnan(v)])) for v in w.to_numpy()]
    return out


def prepare(src, out):
    src, out = Path(src), Path(out)
    boxes = pd.read_csv(src / 'bboxes.csv', index_col=0)
    ref = reference_scores(pd.read_csv(src / 'scores.csv', index_col=0))
    split = {int(p): s for s, ps in json.loads((src / 'train_val_test_split.json').read_text()).items() for p in ps}
    boxes['stem'] = [JOINTS.get(k) for k in joint_key(boxes)]
    rows = []
    for pid, pb in boxes.groupby('patient_id'):
        img = cv2.imread(str(src / 'jpeg' / f'{pid}.jpeg'), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(src / 'jpeg' / f'{pid}.jpeg')
        for hand, hb in pb.groupby('hand'):
            centre = {k: np.array([r.xcenter, r.ycenter], float) for k, r in zip(joint_key(hb), hb.itertuples())}
            scale = REF_PHALANX / np.linalg.norm(centre[('MCP', 3)] - centre[('PIP', 3)])
            himg = hand_image(img, hand)
            case = f'RV_{pid}_{hand[0].upper()}'
            (out / 'rois' / case).mkdir(parents=True, exist_ok=True)
            for r in hb.dropna(subset=['stem']).itertuples():
                size = ROI_SIZE.get(r.stem, FINGER_ROI)
                roi = crop_square(himg, r.xcenter, r.ycenter, int(round(size / scale)))
                path = out / 'rois' / case / f'{r.stem}.png'
                cv2.imwrite(str(path), cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA))
                rows.append({'case': case, 'patient': f'RV_{pid}', 'patient_id': pid, 'hand': hand,
                             'joint': r.stem, 'img_path': str(path), 'scale': scale, 'split': split[pid],
                             'e4_split': 'pool' if split[pid] == 'train' else 'test'})
    m = pd.DataFrame(rows).merge(ref, left_on=['patient_id', 'hand', 'joint'], right_on=['patient_id', 'hand', 'stem'],
                                 how='inner').drop(columns='stem')
    m.to_csv(out / 'manifest.csv', index=False)
    return m


def load_records(manifest, e4_split=None):
    """Records in the format of data.load_records (img_path, case, patient, centre, joint, score)."""
    m = pd.read_csv(manifest)
    if e4_split:
        m = m[m['e4_split'] == e4_split]
    return [{'img_path': r.img_path, 'case': r.case, 'patient': r.patient, 'centre': 'RV', 'joint': r.joint,
             'score': int(r.score)} for r in m.itertuples()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True)
    p.add_argument('--out', default='runs/e4/rheumavit')
    args = p.parse_args(argv)
    m = prepare(args.src, args.out)
    print(f"{len(m)} joints, {m['patient'].nunique()} patients; by E4 split:")
    print(m.groupby('e4_split').agg(patients=('patient', 'nunique'), joints=('joint', 'size')).to_string())
    print('reference grade:', m['score'].value_counts().sort_index().to_dict())
    print('readers per joint:', m['n_readers'].value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
