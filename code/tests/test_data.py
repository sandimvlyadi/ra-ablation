import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ramscore.data import BEDataset, load_records, parse_case, subsample_patients
from ramscore.train import DEFAULT_ROOT


def test_parse_case_qualifies_patient_by_centre():
    c = parse_case('JP_HMCRD_P0050_20210728_6217_L.bmp')
    assert (c.centre, c.patient, c.date, c.side) == ('HMCRD', 'HMCRD_P0050', '20210728', 'L')
    with pytest.raises(ValueError):
        parse_case('bad_name.bmp')


def make_split(root, cases):
    split = root / 'train'
    split.mkdir(parents=True)
    scores = {}
    for case, joint_scores in cases.items():
        (split / case).mkdir()
        for stem, score in joint_scores.items():
            cv2.imwrite(str(split / case / f'{stem}.bmp'), np.full((40, 30), 10 * score, np.uint8))
        scores[f'{case}.bmp'] = {f'BE_{stem}': s for stem, s in joint_scores.items()}
    (split / '_annotation_be_scores.json').write_text(json.dumps(scores))


def test_load_records_and_dataset(tmp_path):
    make_split(tmp_path, {'JP_SARC_P0001_20200101_0001_R': {'MCP-T': 0, 'R': 5},
                          'JP_SARC_P0002_20200101_0002_L': {'PIP-I': 2}})
    records = load_records(tmp_path, 'train')
    assert len(records) == 3
    ds = BEDataset(records, image_size=32)
    item = ds[[r['score'] for r in records].index(5)]
    assert item['img'].shape == (1, 32, 32)
    assert item['label'].item() == 4                   # grade 5 is the 5th class
    assert item['ordinal_target'].tolist() == [1, 1, 1, 1]


def test_subsample_is_patient_level_and_keeps_every_stratum():
    records = [{'patient': f'P{p}', 'score': s} for p in range(100) for s in ([0, 3] if p < 10 else [0, 1])]
    sub = subsample_patients(records, 0.1, seed=0)
    patients = {r['patient'] for r in sub}
    assert len(patients) == 10
    assert any(r['score'] == 3 for r in sub)           # the rare stratum survives
    assert all(sum(r['patient'] == p for r in sub) == 2 for p in patients)


@pytest.mark.skipif(not Path(DEFAULT_ROOT).exists(), reason='RAM-H1200 not available locally')
def test_real_splits_are_patient_disjoint():
    patients = {s: {r['patient'] for r in load_records(DEFAULT_ROOT, s)} for s in ('train', 'val', 'test')}
    assert not patients['train'] & patients['test']
    assert not patients['train'] & patients['val']
    assert not patients['val'] & patients['test']
