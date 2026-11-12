"""RQ1-14 / RQ1-15 normality model after M197 stage 1 (addendum 5)."""
import numpy as np
import pytest
import torch

from ramscore import normality as nm


def test_cutpaste_changes_only_the_masked_region():
    rng = np.random.default_rng(0)
    img = (np.random.default_rng(1).random((64, 64)) * 255).astype(np.uint8)
    out, mask = nm.cutpaste(img, rng)
    assert mask.sum() > 0 and np.array_equal(out[mask == 0], img[mask == 0])
    area = mask.mean()
    assert 0.005 < area < 0.3


def test_patch_targets_mark_overlapping_cells():
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 0:3, 0:3] = 1
    t = nm.patch_targets(mask, (4, 4))
    assert t.shape == (1, 4, 4) and t[0, 0, 0] == 1 and t.sum() == 1


def test_fewsome_loss_prefers_the_right_distances():
    f = torch.randn(2, 8, 4, 4)
    target = torch.zeros(2, 4, 4)
    target[:, 0, 0] = 1
    good = f.clone()
    good[:, :, 0, 0] = -f[:, :, 0, 0]            # pasted cell far (cos -1 -> y_hat clamps at 1), the rest identical
    assert nm.fewsome_loss(f, good, target) < nm.fewsome_loss(f, f, target)


def test_cutpoints_recover_a_separable_score():
    gt = np.repeat([0, 1, 2, 3, 5], 20)
    s = np.repeat([0.1, 0.2, 0.3, 0.4, 0.45], 20) + np.random.default_rng(0).normal(0, 0.005, 100)
    cuts, bacc = nm.choose_cutpoints(s, gt)
    assert bacc == pytest.approx(1.0)
    pred = nm.grades_from_scores(s, cuts)
    assert np.array_equal(nm.level4(pred), nm.level4(gt))


def test_cutpoints_are_ordered_and_use_only_validation_quantiles():
    rng = np.random.default_rng(3)
    s, gt = rng.random(200), rng.choice([0, 1, 2, 3, 5], 200)
    cuts, _ = nm.choose_cutpoints(s, gt)
    assert cuts == sorted(cuts) and all(np.isclose(c, np.quantile(s, np.linspace(0.01, 0.99, 99))).any() for c in cuts)


def test_gate_sets_low_scores_to_zero_and_keeps_the_classifier_otherwise():
    gt = np.array([0] * 10 + [2] * 10)
    s = np.r_[np.full(10, 0.1), np.full(10, 0.9)]
    positive = np.array([3] * 10 + [2] * 10)          # the classifier's grade >= 1
    t, bacc = nm.choose_gate(s, gt, positive)
    assert 0.1 <= t < 0.9 and bacc == pytest.approx(1.0)


def test_pairs_never_pair_a_joint_with_itself():
    imgs = [np.full((32, 32), k, np.uint8) for k in range(3)]
    ds = nm.FewSOMEPairs(imgs, seed=0)
    for i in range(3):
        for _ in range(20):
            item = ds[i]
            clean = item['mask'][0] == 0
            assert not torch.allclose(item['xj'][0][clean], torch.full_like(item['xj'][0][clean], i / 255.0))


def test_patchnet_gives_a_stride_8_map():
    f = nm.PatchNet(pretrained=False)(torch.rand(2, 1, 64, 64))
    assert f.shape == (2, 128, 8, 8)
