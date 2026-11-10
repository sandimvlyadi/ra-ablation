import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from ramscore.data import BEDataset, attach_masks
from ramscore.e6abl import analyse, collapse
from ramscore.models import add_input_channels, build_model
from ramscore.seg import ROOT, UNet, dice_score, from_square, rle_decode, to_square

GRADES = [0, 1, 2, 3, 5]


def rle_encode(mask):
    """COCO compressed RLE of a uint8 mask, to test the decoder without pycocotools."""
    flat = mask.flatten(order='F')
    runs, value, n = [], 0, 0
    for v in flat:
        if v == value:
            n += 1
        else:
            runs.append(n)
            value, n = v, 1
    runs.append(n)
    out = []
    for i, c in enumerate(runs):
        x = int(c) - int(runs[i - 2]) if i > 2 else int(c)
        more = True
        while more:
            b = x & 0x1f
            x >>= 5
            more = not ((x == -1 and b & 0x10) or (x == 0 and not b & 0x10))
            if more:
                b |= 0x20
            out.append(b + 48)
    return bytes(out).decode()


def test_rle_decode_round_trip():
    rng = np.random.default_rng(0)
    mask = (rng.random((40, 30)) > 0.7).astype(np.uint8)
    mask[5:25, 3:20] = 1
    assert (rle_decode(rle_encode(mask), 40, 30) == mask).all()


@pytest.mark.skipif(not Path(ROOT).exists(), reason='dataset not available')
def test_rle_decode_matches_the_annotated_box_on_every_split():
    # 'area' is the box area in some annotations, so the extent of the mask is what can be checked everywhere
    for split in ('train', 'val', 'test'):
        j = json.loads((Path(ROOT) / 'Segmentation' / split / '_annotations_bone_rle.coco.json').read_text())
        for a in j['annotations'][:60]:
            h, w = a['segmentation']['size']
            ys, xs = np.nonzero(rle_decode(a['segmentation']['counts'], h, w))
            x, y, bw, bh = a['bbox']
            assert xs.min() == pytest.approx(x, abs=1) and ys.min() == pytest.approx(y, abs=1)
            assert xs.max() + 1 == pytest.approx(x + bw, abs=1) and ys.max() + 1 == pytest.approx(y + bh, abs=1)


def test_to_square_and_back_keeps_the_mask():
    mask = np.zeros((100, 60), dtype=np.float32)
    mask[20:80, 10:50] = 1
    small, side = to_square(mask, size=64)
    back = from_square(small, 100, 60, side) > 0.5
    assert back.shape == (100, 60)
    assert dice_score(back, mask > 0.5) > 0.9


def test_unet_output_matches_the_input_size():
    out = UNet(pretrained=False)(torch.zeros(2, 1, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_added_channel_starts_at_zero_and_ignores_the_mask():
    net = build_model('resnet34', 5, 'softmax').eval()
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        before = net(x)
        add_input_channels(net)
        after = net(torch.cat([x, torch.rand(2, 1, 64, 64)], 1))
    assert net.conv1.in_channels == 2
    assert torch.allclose(after, before, atol=1e-5)


def test_dataset_is_two_channel_only_with_masks(tmp_path):
    roi, mask = tmp_path / 'r.png', tmp_path / 'test' / 'c' / 'R.png'
    mask.parent.mkdir(parents=True)
    cv2.imwrite(str(roi), np.full((30, 30), 128, dtype=np.uint8))
    cv2.imwrite(str(mask), np.full((30, 30), 255, dtype=np.uint8))
    rec = [{'img_path': str(roi), 'case': 'c', 'patient': 'p', 'centre': 'x', 'joint': 'R', 'score': 1}]
    assert BEDataset(rec, image_size=32)[0]['img'].shape == (1, 32, 32)
    img = BEDataset(attach_masks(rec, tmp_path, 'test'), image_size=32)[0]['img']
    assert img.shape == (2, 32, 32)
    assert img[1].min() == 1


def write(root, name, gt, pred):
    d = root / name
    d.mkdir(parents=True)
    pd.DataFrame({'case': np.repeat([f'h{i:02d}' for i in range(40)], 10), 'patient': np.repeat(np.arange(20), 20),
                  'joint': np.tile([f'J{j}' for j in range(10)], 40), 'gt': gt, 'pred': pred}
                 ).to_csv(d / 'test_predictions.csv', index=False)


def test_analyse_pairs_tags_and_finds_the_better_variant(tmp_path):
    rng = np.random.default_rng(0)
    gt = rng.choice(GRADES, size=400, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    noisy = lambda keep: np.where(rng.random(400) < keep, gt, rng.choice(GRADES, size=400))
    for tag in ('d0_s42', 'd0_s1337', 'd1_s42'):
        write(tmp_path / 'e3', f'e3_a4_resnet34_b2_f0.1_{tag}', gt, noisy(0.3))
    for tag in ('d0_s42', 'd0_s1337'):
        write(tmp_path / 'e6abl', f'e6abl_mask_a4_resnet34_b2_f0.1_{tag}', gt, noisy(0.9))
        write(tmp_path / 'e6abl', f'e6abl_lv3_a4_resnet34_b2_f0.1_{tag}', np.minimum(gt, 2), np.minimum(noisy(0.9), 2))
    res = analyse(tmp_path, n_boot=200)
    assert res['mask'][0]['runs'] == 2 and res['mask'][0]['bacc4']['diff'] > 0 and 'p_holm' in res['mask'][0]['bacc4']
    assert res['levels'][0]['bacc4']['diff'] > 0
    assert res['pool'] == []


def test_collapse_merges_grades_two_and_above():
    df = pd.DataFrame({'gt': [0, 1, 2, 3, 5], 'pred': [5, 3, 2, 1, 0]})
    out = collapse(df)
    assert out['gt'].tolist() == [0, 1, 2, 2, 2] and out['pred'].tolist() == [2, 2, 2, 1, 0]
