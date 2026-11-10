"""RAM-H1200 joint-level bone-erosion (BE) scoring data.

Layout (per split): <root>/<split>/_annotation_be_scores.json and one folder
per hand, <root>/<split>/<case>/<joint>.bmp, where <case> is
<country>_<centre>_P<nnnn>_<YYYYMMDD>_<id>_<L|R>. Dates are shifted per patient.
"""
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# ROI file stem -> score key, as in the official benchmark (datasets/scorecls.py)
BE_JOINTS = {
    'R': 'BE_R', 'U': 'BE_U', 'IP': 'BE_IP', 'L': 'BE_L',
    'MCP-T': 'BE_MCP-T', 'MCP-I': 'BE_MCP-I', 'MCP-M': 'BE_MCP-M', 'MCP-R': 'BE_MCP-R', 'MCP-S': 'BE_MCP-S',
    'CMC-T': 'BE_CMC-T',
    'PIP-I': 'BE_PIP-I', 'PIP-M': 'BE_PIP-M', 'PIP-R': 'BE_PIP-R', 'PIP-S': 'BE_PIP-S',
    'S': 'BE_S', 'Tm': 'BE_Tm',
}
SCORE_FILE = '_annotation_be_scores.json'

# Native SvdH erosion grades present in RAM-H1200 (grade 4 never occurs)
NATIVE_SCORES = (0, 1, 2, 3, 5)
# Primary-endpoint grouping (proposal section 6): 0, 1, 2, >=3
LEVEL4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}


@dataclass(frozen=True)
class Case:
    centre: str
    patient: str     # centre-qualified, e.g. 'HMCRD_P0050'
    date: str        # YYYYMMDD (shifted)
    side: str        # 'L' or 'R'


def parse_case(name):
    parts = Path(name).stem.split('_')
    if len(parts) != 6 or parts[-1] not in ('L', 'R'):
        raise ValueError(f'Unexpected case name: {name}')
    _, centre, pid, date, _, side = parts
    return Case(centre, f'{centre}_{pid}', date, side)


def load_records(root, split):
    """One record per (hand, joint) that has both an ROI image and a score."""
    split_root = Path(root) / split
    scores = json.loads((split_root / SCORE_FILE).read_text())
    records = []
    for image_name, joint_scores in scores.items():
        case_name = Path(image_name).stem
        case = parse_case(case_name)
        for stem, key in BE_JOINTS.items():
            img_path = split_root / case_name / f'{stem}.bmp'
            if key in joint_scores and img_path.exists():
                records.append({
                    'img_path': str(img_path), 'case': case_name, 'patient': case.patient,
                    'centre': case.centre, 'joint': stem, 'score': int(joint_scores[key]),
                })
    if not records:
        raise RuntimeError(f'No BE records under {split_root}')
    return records


AUGMENTATIONS = ('base', 'none', 'geom', 'intensity')


def get_transform(split, aug='base', seed=None):
    """Augmentation of the benchmark's utils.get_cls_transform, minus its leading
    Resize, which BEDataset applies (with the same cv2 bilinear call) before caching.

    aug: base = the benchmark recipe, none / geom / intensity = its ablations (E8). albumentations ignores global
    seeds, so a run is only repeatable with seed set here and seed_augmentation_worker in the DataLoader."""
    if aug not in AUGMENTATIONS:
        raise ValueError(f'Unknown augmentation {aug}; one of {AUGMENTATIONS}')
    if split != 'train' or aug == 'none':
        return None
    steps = []
    if aug in ('base', 'geom'):
        steps.append(A.ShiftScaleRotate(rotate_limit=10))
    if aug in ('base', 'intensity'):
        steps.append(A.RandomBrightnessContrast(p=0.5))
    return A.Compose(steps, seed=seed)


def seed_augmentation_worker(seed, worker_id):
    """DataLoader worker_init_fn: each worker inherits the same Compose state, so without this all workers would draw
    the same random parameters. Use through functools.partial(seed_augmentation_worker, seed)."""
    transform = torch.utils.data.get_worker_info().dataset.transform
    if transform is not None:
        transform.set_random_seed(seed + worker_id)


def read_roi(path, image_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def read_mask(path, image_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def attach_masks(records, mask_root, split):
    """Add the path of each ROI's bone mask (E6 ablation; seg.py writes them), which makes BEDataset 2-channel."""
    return [dict(r, mask_path=str(Path(mask_root) / split / r['case'] / f"{r['joint']}.png")) for r in records]


class BEDataset(Dataset):
    def __init__(self, records, score_values=NATIVE_SCORES, image_size=224, transform=None, preload=True):
        self.records = records
        self.score_values = tuple(score_values)
        self.score_to_class = {s: i for i, s in enumerate(self.score_values)}
        missing = {r['score'] for r in records} - set(self.score_values)
        if missing:
            raise ValueError(f'Scores {sorted(missing)} not in score_values {self.score_values}')
        self.labels = np.array([self.score_to_class[r['score']] for r in records])
        self.image_size = image_size
        self.transform = transform
        # ~50 KB per 224 x 224 ROI, so the whole dataset fits in RAM (~1 GB)
        self.cache = np.stack([read_roi(r['img_path'], image_size) for r in records]) if preload else None
        self.has_mask = bool(records) and 'mask_path' in records[0]
        self.mask_cache = (np.stack([read_mask(r['mask_path'], image_size) for r in records])
                           if preload and self.has_mask else None)

    @property
    def num_classes(self):
        return len(self.score_values)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = self.cache[idx] if self.cache is not None else read_roi(rec['img_path'], self.image_size)
        if self.has_mask:
            mask = self.mask_cache[idx] if self.mask_cache is not None else read_mask(rec['mask_path'], self.image_size)
            if self.transform is not None:
                out = self.transform(image=img, mask=mask)
                img, mask = out['image'], out['mask']
            img = torch.from_numpy(np.stack([img.astype(np.float32) / 255.0, (mask > 127).astype(np.float32)]))
        else:
            if self.transform is not None:
                img = self.transform(image=img)['image']
            img = torch.from_numpy(img.astype(np.float32) / 255.0)[None]
        label = int(self.labels[idx])
        return {
            'img': img,
            'label': torch.tensor(label),
            'ordinal_target': torch.tensor([float(label > k) for k in range(self.num_classes - 1)]),
            'score': torch.tensor(rec['score']),
            'case': rec['case'], 'joint': rec['joint'], 'patient': rec['patient'],
        }


def visit_pairs(records):
    """(earlier, later) record indices for consecutive visits of one joint of one hand (E9). Reads no label, so it can
    take patients outside the labelled draw. Images of one date have no order, so each is paired with the next date."""
    series = {}
    for i, r in enumerate(records):
        c = parse_case(r['case'])
        series.setdefault((r['patient'], c.side, r['joint']), {}).setdefault(c.date, []).append(i)
    pairs = []
    for visits in series.values():
        dates = sorted(visits)
        for early, late in zip(dates, dates[1:]):
            pairs.extend((i, j) for i in visits[early] for j in visits[late])
    return pairs


class VisitPairDataset(Dataset):
    """Images of the visit pairs of a record pool, augmented separately. Holds no scores, so no label can reach the loss."""

    def __init__(self, records, image_size=224, transform=None):
        pairs = visit_pairs(records)
        if not pairs:
            raise ValueError('No visit pairs in these records')
        used = sorted({i for pair in pairs for i in pair})
        position = {i: n for n, i in enumerate(used)}
        self.pairs = np.array([(position[a], position[b]) for a, b in pairs], dtype=np.int64)
        self.cache = np.stack([read_roi(records[i]['img_path'], image_size) for i in used])
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def _image(self, i):
        img = self.cache[i]
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return torch.from_numpy(img.astype(np.float32) / 255.0)[None]

    def __getitem__(self, idx):
        early, late = self.pairs[idx]
        return {'early': self._image(early), 'late': self._image(late)}


def subsample_patients(records, fraction, seed):
    """Keep a patient-level fraction, stratified by each patient's maximum erosion grade.

    Used for the label-fraction experiments (E2); fraction=1.0 returns all records.
    """
    if fraction >= 1.0:
        return list(records)
    by_patient = {}
    for r in records:
        by_patient[r['patient']] = max(by_patient.get(r['patient'], 0), r['score'])
    strata = {}
    for patient, worst in sorted(by_patient.items()):
        strata.setdefault(worst, []).append(patient)
    rng = np.random.default_rng(seed)
    keep = set()
    for patients in strata.values():
        n = max(1, round(fraction * len(patients)))
        keep.update(rng.choice(patients, size=n, replace=False).tolist())
    return [r for r in records if r['patient'] in keep]


def zero_quotas(records):
    """Grade-0 joints to keep per joint type for 1:1 against all other grades (E7).

    Every joint type keeps the same share of its grade-0 joints; the largest remainder makes the total exact.
    """
    zeros = Counter(r['joint'] for r in records if r['score'] == 0)
    n_zero = sum(zeros.values())
    total = min(len(records) - n_zero, n_zero)
    quotas = {j: c * total // n_zero for j, c in sorted(zeros.items())}
    by_remainder = sorted(zeros, key=lambda j: (-(zeros[j] * total % n_zero), j))
    for j in by_remainder[:total - sum(quotas.values())]:
        quotas[j] += 1
    return quotas


def _zeros_by_joint(records):
    pools = {}
    for i, r in enumerate(records):
        if r['score'] == 0:
            pools.setdefault(r['joint'], []).append(i)
    return {j: np.array(pools[j], dtype=np.int64) for j in sorted(pools)}


def undersample_zero(records, seed):
    """Drop grade-0 joints for good, down to zero_quotas; every other joint stays (E7 permanent arm)."""
    rng = np.random.default_rng(seed)
    quotas = zero_quotas(records)
    drop = set()
    for joint, pool in _zeros_by_joint(records).items():
        drop.update(rng.permutation(pool)[quotas[joint]:].tolist())
    return [r for i, r in enumerate(records) if i not in drop]


class RotatingZeroSampler(Sampler):
    """Every epoch: all joints above grade 0 and the next zero_quotas grade-0 joints of each joint type (E7 rotating arm).

    Grade-0 joints are drawn without replacement until a joint type's pool is used up, then from a new permutation, so
    every grade-0 joint is used once per cycle and no joint repeats within an epoch.
    """

    def __init__(self, records, seed):
        self.rng = np.random.default_rng(seed)
        self.quotas = zero_quotas(records)
        self.pools = _zeros_by_joint(records)
        self.queues = {j: self.rng.permutation(pool) for j, pool in self.pools.items()}
        self.always = np.array([i for i, r in enumerate(records) if r['score'] != 0], dtype=np.int64)

    def __len__(self):
        return len(self.always) + sum(self.quotas.values())

    def _take(self, joint):
        n, queue = self.quotas[joint], self.queues[joint]
        head, rest = queue[:n], queue[n:]
        if len(head) < n:
            # the leftovers open this epoch, so they close the new cycle and nothing repeats within the epoch
            fresh = self.rng.permutation(self.pools[joint])
            fresh = fresh[~np.isin(fresh, head)]
            missing = n - len(head)
            head, rest = np.concatenate([head, fresh[:missing]]), np.concatenate([fresh[missing:], head])
        self.queues[joint] = rest
        return head

    def __iter__(self):
        zeros = [self._take(j) for j in self.quotas]
        return iter(self.rng.permutation(np.concatenate([self.always, *zeros])).tolist())


def _composition(records):
    zero = [r for r in records if r['score'] == 0]
    return {'total': len(records), 'zero': len(zero), 'nonzero': len(records) - len(zero),
            'patients': len({r['patient'] for r in records}),
            'zero_per_joint': dict(sorted(Counter(r['joint'] for r in zero).items())),
            'nonzero_per_joint': dict(sorted(Counter(r['joint'] for r in records if r['score'] != 0).items())),
            'zero_per_centre': dict(sorted(Counter(r['centre'] for r in zero).items())),
            'nonzero_per_centre': dict(sorted(Counter(r['centre'] for r in records if r['score'] != 0).items()))}


def undersample_summary(records, mode, seed):
    """Composition of an E7 run, written to undersample.json. records: the train split before any trimming."""
    pool = _composition(records)
    quotas = zero_quotas(records)
    n_zero = sum(quotas.values())
    return {'mode': mode, 'seed': seed, 'pool': pool, 'zero_quota': quotas,
            'per_epoch': {'zero': n_zero, 'nonzero': pool['nonzero'], 'total': n_zero + pool['nonzero']},
            'kept': _composition(undersample_zero(records, seed)) if mode == 'permanent' else None}
