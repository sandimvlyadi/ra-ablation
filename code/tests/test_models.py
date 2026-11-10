import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ramscore.models import (HEADS, Criterion, build_model, corn_loss, decode, freeze_backbone, head_dims,
                             head_priors, load_backbone)

K = 5
LABELS = np.array([0] * 70 + [1] * 15 + [2] * 9 + [3] * 2 + [4] * 4)


def test_ordinal_loss_matches_benchmark_bce():
    torch.manual_seed(0)
    logits, labels = torch.randn(8, K - 1), torch.tensor([0, 1, 2, 3, 4, 0, 2, 4])
    target = torch.stack([(labels > k).float() for k in range(K - 1)], dim=1)
    assert torch.allclose(Criterion('ordinal', LABELS, K)(logits, labels),
                          F.binary_cross_entropy_with_logits(logits, target))
    pos = np.array([(LABELS > k).sum() for k in range(K - 1)])
    pos_weight = torch.tensor((len(LABELS) - pos) / pos, dtype=torch.float32)
    assert torch.allclose(Criterion('ordinal', LABELS, K, class_weight=True)(logits, labels),
                          F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight))


def test_softmax_logit_adjustment_is_balanced_softmax():
    torch.manual_seed(0)
    logits, labels = torch.randn(8, K), torch.tensor([0, 1, 2, 3, 4, 0, 0, 1])
    prior = torch.tensor(np.bincount(LABELS) / len(LABELS), dtype=torch.float32)
    assert torch.allclose(Criterion('softmax', LABELS, K, logit_adjust=1.0)(logits, labels),
                          F.cross_entropy(logits + prior.log(), labels))
    assert torch.allclose(Criterion('softmax', LABELS, K)(logits, labels), F.cross_entropy(logits, labels))


def test_corn_loss_uses_conditional_subsets():
    logits, labels = torch.zeros(3, K - 1), torch.tensor([0, 2, 4])
    # task k sees labels >= k: 3 + 2 + 2 + 1 = 8 terms, each log(2) at zero logits
    assert corn_loss(logits, labels).item() == pytest.approx(np.log(2))
    perfect = torch.tensor([[-9.0] * 4, [9.0, 9.0, -9.0, 0.0], [9.0] * 4])
    assert corn_loss(perfect, labels).item() < 1e-3


def test_priors_are_on_each_output_scale():
    corn = head_priors('corn', LABELS, K)
    assert corn[0] == pytest.approx(np.log(30 / 70))            # P(y > 0) = 0.30
    assert corn[1] == pytest.approx(np.log(15 / 15))            # P(y > 1 | y >= 1) = 15 / 30
    hier = head_priors('hier', LABELS, K)
    assert hier[0] == pytest.approx(np.log(30 / 70))
    assert np.exp(hier[1:]).sum() == pytest.approx(1.0)


def test_decoding():
    big = 9.0
    corn = torch.tensor([[big, big, -big, big], [-big, big, big, big]])
    assert decode(corn, 'corn').tolist() == [2, 0]              # cumulative product stops at the first "no"
    hier = torch.tensor([[-1.0, 5, 0, 0, 0], [1.0, 0, 0, 5, 0]])
    assert decode(hier, 'hier').tolist() == [0, 3]
    assert decode(torch.tensor([[big, -big, big, -big]]), 'ordinal').tolist() == [2]


@pytest.mark.parametrize('head', HEADS)
def test_every_head_trains_on_class_indices(head):
    torch.manual_seed(0)
    logits = torch.randn(16, head_dims(head, K), requires_grad=True)
    loss = Criterion(head, LABELS, K, logit_adjust=1.0)(logits, torch.randint(0, K, (16,)))
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() > 0


def test_class_weight_rejected_where_undefined():
    with pytest.raises(ValueError):
        Criterion('corn', LABELS, K, class_weight=True)


def test_load_backbone_resets_classifier_and_freezes(tmp_path):
    src = build_model('resnet18', K, 'ordinal')
    torch.save({'model': src.state_dict()}, tmp_path / 'ckpt.pt')
    dst = build_model('resnet18', K, 'softmax')                  # different head size still loads
    assert set(load_backbone(dst, tmp_path / 'ckpt.pt')) == {'fc.weight', 'fc.bias'}
    assert torch.equal(dst.conv1.weight, src.conv1.weight)
    freeze_backbone(dst)
    trainable = {n for n, p in dst.named_parameters() if p.requires_grad}
    assert trainable == {'fc.weight', 'fc.bias'}
