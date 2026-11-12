"""RQ1-07 bone age pre-training helpers (addendum 6)."""
import io
import zipfile

import cv2
import numpy as np
import pandas as pd

from ramscore import boneage as ba


def test_pad_square_keeps_the_image_centred():
    img = np.full((4, 2), 7, np.uint8)
    out = ba.pad_square(img)
    assert out.shape == (4, 4) and (out[:, 1:3] == 7).all() and (out[:, [0, 3]] == 0).all()


def test_split_is_seeded_disjoint_and_complete():
    ids = list(range(100))
    tr, va = ba.split_ids(ids, 2026)
    assert len(va) == 10 and not set(tr) & set(va) and sorted(tr + va) == ids
    assert ba.split_ids(ids[::-1], 2026) == (tr, va)


def test_prepare_reads_the_kaggle_layout(tmp_path):
    zpath = tmp_path / 'rsna.zip'
    with zipfile.ZipFile(zpath, 'w') as z:
        z.writestr('boneage-training-dataset.csv', 'id,boneage,male\n1,120,True\n2,60,False\n3,90,True\n')
        for i, (h, w) in {1: (60, 40), 2: (30, 50)}.items():
            z.writestr(f'boneage-training-dataset/boneage-training-dataset/{i}.png',
                       cv2.imencode('.png', np.full((h, w), 100, np.uint8))[1].tobytes())
    ba.prepare(zpath, tmp_path / 'out', size=32)
    labels = pd.read_csv(tmp_path / 'out' / 'labels.csv')
    assert list(labels['id']) == [1, 2]          # id 3 has no image
    img = cv2.imread(str(tmp_path / 'out' / 'images' / '1.png'), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (32, 32) and img[:, 0].max() == 0 and img[16, 16] == 100
    assert len((tmp_path / 'out' / 'SHA256SUMS').read_text().split('\n')) == 3
