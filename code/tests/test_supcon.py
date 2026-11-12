"""SupCon + CE (RQ1-12) and grade-distance weighted negatives (RQ1-13, after M219)."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ramscore.data import BEDataset, get_transform
from ramscore.models import ProjectionHead, build_model, pooled_and_logits, supcon_loss
from ramscore.train import check_supcon_args, get_args


def manual_supcon(z, y, t, alpha=None, c=5):
    """Direct transcription of Khosla L_out with the M219 weight in the denominator."""
    n, total, anchors = len(y), 0.0, 0
    for i in range(n):
        pos = [p for p in range(n) if p != i and y[p] == y[i]]
        if not pos:
            continue
        den = 0.0
        for k in range(n):
            if k == i:
                continue
            r = 1.0 if y[k] == y[i] or alpha is None else alpha + (1 - alpha) * abs(y[i] - y[k]) / (c - 1)
            den += r * math.exp(float(z[i] @ z[k]) / t)
        total += -sum(float(z[i] @ z[p]) / t - math.log(den) for p in pos) / len(pos)
        anchors += 1
    return total / anchors


@pytest.mark.parametrize('alpha', [None, 0.25, 0.6])
def test_matches_the_formula(alpha):
    g = torch.Generator().manual_seed(0)
    z = F.normalize(torch.randn(10, 8, generator=g), dim=1)
    y = torch.tensor([0, 0, 1, 1, 2, 4, 4, 0, 3, 1])
    got = supcon_loss(z, y, 0.1, alpha)
    assert got.item() == pytest.approx(manual_supcon(z, y.tolist(), 0.1, alpha), rel=1e-5)


def test_alpha_one_is_plain_supcon():
    z = F.normalize(torch.randn(12, 16), dim=1)
    y = torch.randint(0, 5, (12,))
    assert supcon_loss(z, y, 0.1, 1.0).item() == pytest.approx(supcon_loss(z, y, 0.1).item(), rel=1e-6)


def test_weighting_pushes_distant_grades_harder():
    # anchor 0 (grade 0) and its positive; one negative of grade 1 and one of grade 4, both equally similar
    z = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.5, 0.5], [0.5, 0.5]]), dim=1)
    y = torch.tensor([0, 0, 1, 4])
    near = torch.tensor([0, 0, 1, 1])   # same embeddings, both negatives adjacent
    weighted = supcon_loss(z, y, 0.1, 0.25)
    assert weighted.item() > supcon_loss(z, near, 0.1, 0.25).item()


def test_anchors_without_positives_are_left_out():
    z = F.normalize(torch.randn(4, 8), dim=1)
    assert supcon_loss(z, torch.tensor([0, 1, 2, 3]), 0.1).item() == 0.0
    y = torch.tensor([0, 0, 1, 2])
    assert supcon_loss(z, y, 0.1).item() == pytest.approx(manual_supcon(z, y.tolist(), 0.1), rel=1e-5)


def test_clustered_embeddings_score_lower_and_gradients_flow():
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    clustered = F.one_hot(y, 8).float()
    random = torch.randn(6, 8, requires_grad=True)
    loss = supcon_loss(F.normalize(random, dim=1), y, 0.1)
    loss.backward()
    assert supcon_loss(clustered, y, 0.1).item() < loss.item() and random.grad.abs().sum() > 0


def test_projection_is_unit_norm_and_pooled_logits_match_forward():
    net = build_model('resnet34', 5, 'softmax').eval()
    x = torch.randn(2, 1, 64, 64)
    feats, logits = pooled_and_logits(net, x)
    assert feats.shape == (2, net.num_features) and torch.allclose(logits, net(x), atol=1e-5)
    z = ProjectionHead(net.num_features)(feats)
    assert z.shape == (2, 128) and torch.allclose(z.norm(dim=1), torch.ones(2), atol=1e-5)


def test_two_views_are_independent_draws(tmp_path):
    import cv2
    recs = []
    for i in range(2):
        path = tmp_path / f'r{i}.png'
        cv2.imwrite(str(path), (np.random.default_rng(i).random((64, 64)) * 255).astype(np.uint8))
        recs.append({'img_path': str(path), 'score': 0, 'case': f'c{i}', 'joint': 'MCP-I', 'patient': 'P0'})
    ds = BEDataset(recs, image_size=64, transform=get_transform('train', 'base', seed=1))
    assert 'img2' not in ds[0]
    ds.views = 2
    item = ds[0]
    assert item['img2'].shape == item['img'].shape and not torch.equal(item['img'], item['img2'])
    plain = BEDataset(recs, image_size=64, transform=None)
    plain.views = 2
    assert torch.equal(plain[1]['img'], plain[1]['img2'])


BASE = ['--name', 'x', '--head', 'softmax']


@pytest.mark.parametrize('extra', [[], ['--supcon-weight', '0.5'], ['--supcon-weight', '1', '--supcon-alpha', '0.25'],
                                   ['--supcon-weight', '1', '--pretrained', '--logit-adjust', '1']])
def test_valid_supcon_combinations_pass(extra):
    check_supcon_args(get_args(BASE + extra))


@pytest.mark.parametrize('extra', [['--supcon-alpha', '0.25'], ['--supcon-weight', '1', '--supcon-alpha', '0'],
                                   ['--supcon-weight', '1', '--supcon-alpha', '1.5'],
                                   ['--supcon-weight', '1', '--pair-weight', '0.1'],
                                   ['--supcon-weight', '1', '--layer-decay', '0.85'],
                                   ['--supcon-weight', '1', '--pretrained', '--freeze-backbone']])
def test_invalid_supcon_combinations_are_refused(extra):
    with pytest.raises(SystemExit):
        check_supcon_args(get_args(BASE + extra))
