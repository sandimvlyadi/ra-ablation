"""RQ1-17 flags: linear probe from ImageNet weights (--pretrained --freeze-backbone) and LP-FT stage 2 (--keep-head)."""
import pytest
import torch

from ramscore.models import build_model, load_full
from ramscore.train import check_init_args, get_args

K = 5


def args(*extra):
    return get_args(['--name', 'x', *extra])


def test_defaults_keep_the_old_recipe():
    a = args()
    assert a.keep_head is False and a.freeze_backbone is False
    check_init_args(a)


@pytest.mark.parametrize('extra', [('--pretrained', '--freeze-backbone'), ('--init-from', 'e.pt', '--freeze-backbone'),
                                   ('--init-from', 'lp.pt', '--keep-head')])
def test_valid_combinations_pass(extra):
    check_init_args(args(*extra))


@pytest.mark.parametrize('extra', [('--freeze-backbone',), ('--keep-head',), ('--pretrained', '--keep-head'),
                                   ('--init-from', 'lp.pt', '--keep-head', '--freeze-backbone'),
                                   ('--init-from', 'lp.pt', '--keep-head', '--mask-root', 'm')])
def test_invalid_combinations_are_refused(extra):
    with pytest.raises(SystemExit):
        check_init_args(args(*extra))


def test_load_full_keeps_the_trained_head(tmp_path):
    src = build_model('resnet18', K, 'softmax')
    with torch.no_grad():
        src.fc.weight.fill_(0.5)
    torch.save({'model': src.state_dict(), 'epoch': 3}, tmp_path / 'lp.pt')
    dst = build_model('resnet18', K, 'softmax')
    load_full(dst, tmp_path / 'lp.pt')
    assert torch.equal(dst.fc.weight, src.fc.weight) and torch.equal(dst.conv1.weight, src.conv1.weight)


def test_load_full_refuses_another_head(tmp_path):
    torch.save({'model': build_model('resnet18', K, 'ordinal').state_dict()}, tmp_path / 'lp.pt')   # K-1 outputs
    with pytest.raises(RuntimeError):
        load_full(build_model('resnet18', K, 'softmax'), tmp_path / 'lp.pt')
