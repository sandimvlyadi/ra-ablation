import operator
from pathlib import Path

import numpy as np
import pytest
import torch

from ramscore.models import build_model, load_backbone
from ramscore.pool import H1200_ROOT, GroupBatchSampler, chrono_groups, h1200_patients, load_pool
from ramscore.ssl import DINO, MAE, MultiCrop, chronocon_loss, cosine_schedule


def chronocon_reference(f, g, t, tau=1.0):
    """Eq. 2-3 of Watzenboeck et al. (2026), written as explicit loops."""
    s = -torch.cdist(f, f) / tau
    n = len(f)

    def part(pair_ok, beyond):
        terms = []
        for a in range(n):
            for p in range(n):
                if a == p or g[a] != g[p] or not pair_ok(t[a], t[p]):
                    continue
                negs = [k for k in range(n) if k != a and g[k] == g[a] and beyond(t[p], t[k])]
                if negs:
                    denom = torch.logsumexp(torch.stack([s[a, p]] + [s[a, k] for k in negs]), 0)
                    terms.append(denom - s[a, p])
        return torch.stack(terms).mean() if terms else torch.tensor(0.0)

    return part(operator.le, operator.lt) + part(operator.ge, operator.gt)


def test_chronocon_matches_reference():
    torch.manual_seed(0)
    g = torch.tensor([0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 2])          # 3 series, two crops each
    t = torch.tensor([1, 2, 5, 3, 3, 7, 1, 2, 5, 3, 3, 7])          # series 1 has two images on one date
    f = torch.randn(12, 8)
    assert torch.allclose(chronocon_loss(f, g, t), chronocon_reference(f, g, t), atol=1e-5)


def test_chronocon_prefers_chronological_embeddings():
    g, t = torch.zeros(8, dtype=torch.long), torch.arange(4).repeat(2)
    ordered = t[:, None].float() * torch.tensor([[1.0, 0.0]])
    shuffled = torch.tensor([2, 0, 3, 1]).repeat(2)[:, None].float() * torch.tensor([[1.0, 0.0]])
    assert chronocon_loss(ordered, g, t) < chronocon_loss(shuffled, g, t)


def test_chronocon_ignores_series_without_negatives():
    f = torch.randn(4, 8, requires_grad=True)
    loss = chronocon_loss(f, torch.tensor([0, 1, 2, 3]), torch.tensor([1, 1, 1, 1]))
    assert loss.item() == 0.0


def test_chrono_groups_and_sampler():
    records = [{'patient': p, 'side': 'L', 'roi': 'BE/MCP-I', 'time': t}
               for p, times in [('A', [1, 2, 3, 4, 5]), ('B', [1, 1]), ('C', [2, 9])] for t in times]
    groups = chrono_groups(records)
    assert sorted(len(x) for x in groups) == [2, 5]                  # B has one date only
    batches = list(GroupBatchSampler(groups, groups_per_batch=2, max_visits=3, seed=0))
    assert len(batches) == 1 and len(batches[0]) == 5                # 3 of A (capped) + 2 of C


def test_cosine_schedule():
    s = cosine_schedule(1.0, 0.0, 100, warmup_steps=10)
    assert len(s) == 100 and s[0] == 0.0 and s[10] == pytest.approx(1.0) and s[-1] < 1e-3


def test_mae_reconstructs_and_encoder_transfers(tmp_path):
    torch.manual_seed(0)
    mae = MAE('vit_tiny_patch16_224')
    loss = mae(torch.rand(2, 1, 224, 224))
    loss.backward()
    assert torch.isfinite(loss) and mae.backbone.patch_embed.proj.weight.grad is not None
    _, mask, _ = mae.encode(torch.rand(2, 1, 224, 224))
    assert mask.mean().item() == pytest.approx(0.75, abs=0.01)
    torch.save({'model': mae.backbone.state_dict()}, tmp_path / 'enc.pt')
    net = build_model('vit_tiny_patch16_224', 5, 'ordinal')
    assert set(load_backbone(net, tmp_path / 'enc.pt')) == {'head.weight', 'head.bias'}


def test_dino_multicrop_step_and_teacher_transfer(tmp_path):
    torch.manual_seed(0)
    dino = DINO('resnet18', out_dim=64)
    crops = MultiCrop(n_local=2)
    per_image = [crops(np.random.randint(0, 255, (120, 110), np.uint8)) for _ in range(3)]
    views = [torch.stack(v) for v in zip(*per_image)]
    assert [tuple(v.shape[-2:]) for v in views] == [(224, 224)] * 2 + [(96, 96)] * 2
    loss = dino(views, teacher_temp=0.04)
    loss.backward()
    before = dino.teacher_backbone.conv1.weight.clone()
    torch.optim.SGD(dino.student_parameters(), lr=1.0).step()
    dino.update_teacher(0.5)
    assert torch.isfinite(loss) and not torch.equal(before, dino.teacher_backbone.conv1.weight)
    assert all(not p.requires_grad for p in dino.teacher_parameters())
    torch.save({'model': dino.teacher_backbone.state_dict()}, tmp_path / 'enc.pt')
    net = build_model('resnet18', 5, 'corn')
    assert set(load_backbone(net, tmp_path / 'enc.pt')) == {'fc.weight', 'fc.bias'}


@pytest.mark.skipif(not Path(H1200_ROOT).exists(), reason='RAM-H1200 not available locally')
def test_real_pool_excludes_test_patients():
    pool = load_pool()
    assert not {r['patient'] for r in pool} & h1200_patients(H1200_ROOT, 'test')
    assert {r['roi'].split('/')[0] for r in pool} == {'BE', 'JSN'}
