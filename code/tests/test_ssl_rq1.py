"""SimCLR (RQ1-08) and BYOL (RQ1-10) objectives, addendum 4."""
import math

import pytest
import torch
import torch.nn.functional as F

from ramscore.models import build_model, load_backbone
from ramscore.pretrain import DEFAULTS, get_args
from ramscore.ssl import BYOL, SimCLR, byol_regression, nt_xent


def manual_nt_xent(z, t):
    z = F.normalize(z, dim=1)
    n2 = len(z)
    n = n2 // 2
    total = 0.0
    for i in range(n2):
        j = (i + n) % n2
        den = sum(math.exp(float(z[i] @ z[k]) / t) for k in range(n2) if k != i)
        total += -(float(z[i] @ z[j]) / t - math.log(den))
    return total / n2


def test_nt_xent_matches_the_formula():
    z = torch.randn(8, 5, generator=torch.Generator().manual_seed(1))
    assert nt_xent(z, 0.1).item() == pytest.approx(manual_nt_xent(z, 0.1), rel=1e-5)


def test_nt_xent_is_low_when_views_agree():
    base = F.one_hot(torch.arange(4), 8).float()
    agree = torch.cat([base, base])
    swapped = torch.cat([base, base.roll(1, dims=0)])
    assert nt_xent(agree, 0.1).item() < 0.01 < nt_xent(swapped, 0.1).item()


def test_byol_regression_range():
    p = torch.randn(6, 4)
    assert byol_regression(p, p).item() == pytest.approx(0.0, abs=1e-6)
    assert byol_regression(p, -p).item() == pytest.approx(4.0, abs=1e-5)


def views(b=4, size=64):
    g = torch.Generator().manual_seed(0)
    return [torch.rand(b, 1, size, size, generator=g), torch.rand(b, 1, size, size, generator=g)]


def test_simclr_backward_and_encoder_loads_into_the_classifier(tmp_path):
    model = SimCLR('resnet18')
    loss = model(views())
    loss.backward()
    assert torch.isfinite(loss) and model.backbone.conv1.weight.grad is not None
    net = build_model('resnet18', 5, 'softmax')
    assert load_backbone(net, _save(model.backbone, tmp_path)) == ['fc.weight', 'fc.bias']


def _save(backbone, folder):
    path = folder / 'encoder.pt'
    torch.save({'model': backbone.state_dict()}, path)
    return path


def test_byol_target_has_no_gradient_and_follows_the_online_network():
    model = BYOL('resnet18', hidden=64, proj_dim=16)
    loss = model(views())
    loss.backward()
    assert all(p.grad is None for p in model.teacher_parameters())
    assert model.predictor[0].weight.grad is not None
    with torch.no_grad():
        for p in model.backbone.parameters():
            p.add_(1.0)
    before = [t.clone() for t in model.target_backbone.parameters()]
    model.update_teacher(0.9)
    moved = [(t - b).abs().sum().item() for t, b in zip(model.target_backbone.parameters(), before)]
    assert all(m > 0 for m in moved)
    model.update_teacher(1.0)
    after = [t.clone() for t in model.target_backbone.parameters()]
    model.update_teacher(1.0)
    assert all(torch.equal(a, t) for a, t in zip(after, model.target_backbone.parameters()))


def test_defaults_follow_the_e1_dino_recipe():
    for method in ('simclr', 'byol'):
        a = get_args(['--name', 'x', '--method', method])
        d = DEFAULTS['dino']
        assert (a.epochs, a.lr, a.weight_decay, a.weight_decay_end, a.warmup_epochs) == (
            d['epochs'], d['lr'], d['weight_decay'], d['weight_decay_end'], d['warmup_epochs'])
        assert a.batch_size == 128 and a.temperature == 0.1


def test_chronocon_is_refused_for_the_new_methods():
    with pytest.raises(SystemExit):
        get_args(['--name', 'x', '--method', 'byol', '--chrono-weight', '1'])
