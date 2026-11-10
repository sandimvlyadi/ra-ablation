"""Unlabelled joint-ROI pool for self-supervised pre-training (proposal A3-A5).

The main pool is every BE and JSN ROI of the RAM-H1200 train and val splits.
Test patients are excluded explicitly (anti-leakage rule, proposal 5.2), even
though the splits are already patient-disjoint. RAM-W600 wrist ROIs can be added
(--pool h1200+w600) for the pool ablation in E6; its patient ids are coded
differently, so overlap with RAM-H1200 test patients cannot be checked.

Chronological groups (A5) are (patient, side, ROI type) series with at least two
distinct visit dates, as in ChronoCon (Watzenboeck et al., 2026).
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .data import parse_case

H1200_ROOT = '/data/Smart-X/tesis-fix/dataset/RAM-H1200-v1'
W600_ROOT = '/data/Smart-X/tesis-fix/dataset/RAM-W600'
H1200_TASKS = {'BE': 'SvdH_BE_Scoring', 'JSN': 'SvdH_JSN_Scoring'}
W600_TASK = 'SvdHBEScoreClassification'


def _cases(split_root):
    return sorted(p for p in Path(split_root).iterdir() if p.is_dir())


def h1200_patients(root, split):
    return {parse_case(c.name).patient for task in H1200_TASKS.values()
            for c in _cases(Path(root) / 'SvdH_Scoring' / task / split)}


def load_pool(h1200_root=H1200_ROOT, splits=('train', 'val'), tasks=('BE', 'JSN'), w600_root=None):
    """One record per ROI image: path, patient, side, ROI type ('BE/MCP-I'), visit time (int)."""
    excluded = h1200_patients(h1200_root, 'test')
    records = []
    for task in tasks:
        for split in splits:
            for case_dir in _cases(Path(h1200_root) / 'SvdH_Scoring' / H1200_TASKS[task] / split):
                case = parse_case(case_dir.name)
                if case.patient in excluded:
                    continue
                records += [{'img_path': str(p), 'dataset': 'H1200', 'patient': case.patient, 'side': case.side,
                             'roi': f'{task}/{p.stem}', 'time': int(case.date)}
                            for p in sorted(case_dir.glob('*.bmp'))]
    if w600_root:
        for split in splits:
            for case_dir in _cases(Path(w600_root) / W600_TASK / split):
                patient, visit, side = case_dir.name.split('_')      # e.g. 0001_0001_L
                records += [{'img_path': str(p), 'dataset': 'W600', 'patient': f'W600_{patient}', 'side': side,
                             'roi': f'W600/{p.stem}', 'time': int(visit)}
                            for p in sorted(case_dir.glob('*.bmp'))]
    if not records:
        raise RuntimeError(f'No ROIs found under {h1200_root}')
    return records


def chrono_groups(records):
    """Indices of each (patient, side, ROI) series that has >= 2 distinct visit times."""
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault((r['patient'], r['side'], r['roi']), []).append(i)
    return [idx for idx in groups.values() if len({records[i]['time'] for i in idx}) >= 2]


class PoolDataset(Dataset):
    """ROIs cached at native size (grayscale uint8); `views` maps one image to a list of tensors."""

    def __init__(self, records, views):
        self.records = records
        self.views = views
        self.images = [cv2.imread(r['img_path'], cv2.IMREAD_GRAYSCALE) for r in records]
        bad = [r['img_path'] for r, im in zip(records, self.images) if im is None]
        if bad:
            raise FileNotFoundError(f'{len(bad)} unreadable ROIs, e.g. {bad[0]}')
        self.group = np.full(len(records), -1)
        self.time = np.array([r['time'] for r in records])

    def set_groups(self, groups):
        for g, idx in enumerate(groups):
            self.group[idx] = g

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return {'views': self.views(self.images[idx]), 'group': torch.tensor(self.group[idx]),
                'time': torch.tensor(self.time[idx])}


class GroupBatchSampler(Sampler):
    """Batches of whole chronological series: `groups_per_batch` series, up to `max_visits` images each."""

    def __init__(self, groups, groups_per_batch, max_visits, seed=0):
        self.groups, self.k, self.max_visits = groups, groups_per_batch, max_visits
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.groups) // self.k

    def __iter__(self):
        order = self.rng.permutation(len(self.groups))
        for b in range(len(self)):
            batch = []
            for g in order[b * self.k:(b + 1) * self.k]:
                idx = self.groups[g]
                if len(idx) > self.max_visits:
                    idx = self.rng.choice(idx, self.max_visits, replace=False).tolist()
                batch += idx
            yield batch
