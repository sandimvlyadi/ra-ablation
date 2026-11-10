"""Backbones, output heads and their losses (proposal factor B).

Heads (K = number of grades):
- 'ordinal' (B0, B1): the benchmark head. K-1 independent threshold logits for
  P(y > k), trained with BCE, decoded as the number of thresholds passed.
- 'softmax' (B2): a plain K-way classifier with cross-entropy.
- 'corn' (B3, B4): CORN (Shi et al., 2023). K-1 conditional logits for
  P(y > k | y >= k), task k trained only on joints with y >= k, decoded from
  the cumulative product, so predictions are always rank-consistent.
- 'hier' (B5): logit 0 for 0 vs > 0, then a (K-1)-way softmax over the grades
  of positive joints, trained only on positive joints.

Logit adjustment (Menon et al., 2021) adds tau * log prior to the logits in the
loss only; inference uses the raw logits. For the softmax head with tau = 1 this
is the balanced softmax of Ren et al. (2020). For binary tasks (ordinal and CORN
thresholds, the 0 vs > 0 gate) the prior enters as its log-odds.
"""
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn

HEADS = ('ordinal', 'softmax', 'corn', 'hier')

# Benchmark model names (main_score_cls.py) -> timm names
ALIASES = {
    'ResNet18': 'resnet18', 'ResNet34': 'resnet34', 'ResNet50': 'resnet50',
    'MobileNet': 'mobilenetv2_050', 'EfficientFormer': 'efficientformerv2_s0',
    'EfficientNetV2': 'tf_efficientnetv2_s.in21k_ft_in1k', 'MobileViT': 'mobilevit_s',
    'LeViT': 'levit_128s', 'ConvNeXtV2': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
}


def head_dims(head, num_classes):
    return num_classes - 1 if head in ('ordinal', 'corn') else num_classes


def build_model(name, num_classes, head='ordinal', pretrained=False, in_chans=1):
    return timm.create_model(ALIASES.get(name, name), pretrained=pretrained, in_chans=in_chans,
                             num_classes=head_dims(head, num_classes))


def log_odds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def head_priors(head, labels, num_classes):
    """Training-set prior of every output, on the logit scale of that output."""
    labels = np.asarray(labels)
    counts = np.maximum(np.bincount(labels, minlength=num_classes), 1)
    thresholds = range(num_classes - 1)
    if head == 'ordinal':
        return log_odds(np.array([(labels > k).mean() for k in thresholds]))
    if head == 'corn':
        return log_odds(np.array([(labels > k).sum() / max((labels >= k).sum(), 1) for k in thresholds]))
    if head == 'softmax':
        return np.log(counts / counts.sum())
    positive = counts[1:]
    return np.concatenate([[log_odds((labels > 0).mean())], np.log(positive / positive.sum())])


def corn_loss(logits, labels):
    """CORN loss as in coral_pytorch: BCE summed over each task's subset, divided by the subset sizes."""
    total, n = logits.new_zeros(()), 0
    for k in range(logits.shape[1]):
        subset = labels >= k
        if subset.any():
            total = total + F.binary_cross_entropy_with_logits(
                logits[subset, k], (labels[subset] > k).float(), reduction='sum')
            n += int(subset.sum())
    return total / max(n, 1)


class Criterion(nn.Module):
    """Loss on (logits, class index) for any head, with optional class weights and logit adjustment."""

    def __init__(self, head, train_labels, num_classes, class_weight=False, logit_adjust=0.0):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f'unknown head {head!r}')
        if class_weight and head not in ('ordinal', 'softmax'):
            raise ValueError(f'class weights are only defined for the ordinal and softmax heads, not {head!r}')
        labels = np.asarray(train_labels)
        self.head, self.num_classes = head, num_classes
        offset = logit_adjust * head_priors(head, labels, num_classes)
        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float32))
        weight = None
        if class_weight and head == 'ordinal':
            # Benchmark --use_class_weight: per-threshold neg/pos ratio
            pos = np.array([(labels > k).sum() for k in range(num_classes - 1)])
            weight = (len(labels) - pos) / np.maximum(pos, 1)
        elif class_weight:
            counts = np.bincount(labels, minlength=num_classes)
            weight = len(labels) / (num_classes * np.maximum(counts, 1))
        self.register_buffer('weight', None if weight is None else torch.tensor(weight, dtype=torch.float32))

    def forward(self, logits, labels):
        z = logits + self.offset
        if self.head == 'ordinal':
            target = (labels[:, None] > torch.arange(self.num_classes - 1, device=labels.device)).float()
            return F.binary_cross_entropy_with_logits(z, target, pos_weight=self.weight)
        if self.head == 'softmax':
            return F.cross_entropy(z, labels, weight=self.weight)
        if self.head == 'corn':
            return corn_loss(z, labels)
        positive = labels > 0
        loss = F.binary_cross_entropy_with_logits(z[:, 0], positive.float())
        if positive.any():
            loss = loss + F.cross_entropy(z[positive, 1:], labels[positive] - 1)
        return loss


def decode(logits, head):
    if head == 'ordinal':
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)
    if head == 'corn':
        return (torch.cumprod(torch.sigmoid(logits), dim=1) > 0.5).sum(dim=1)
    if head == 'hier':
        grade = logits[:, 1:].argmax(dim=1) + 1
        return torch.where(logits[:, 0] > 0, grade, torch.zeros_like(grade))
    return logits.argmax(dim=1)


def expected_grade(logits):
    """Mean class position under the softmax, as in QWK: grade 5 sits one step above grade 3."""
    positions = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
    return (F.softmax(logits, dim=1) * positions).sum(dim=1)


def monotonic_penalty(early, late, margin=0.0):
    """Hinge on a fall of the expected grade between two visits of one joint (E9). The margin leaves room for real
    repair and reading error. Takes the raw logits, the ones inference decodes."""
    return F.relu(expected_grade(early) - expected_grade(late) - margin).mean()


def load_backbone(net, path):
    """Load encoder weights from a checkpoint, leaving the classifier freshly initialised.

    Accepts this project's checkpoints ({'model': state_dict}) or a bare state dict.
    Returns the names of the classifier parameters that were left out.
    """
    state = torch.load(path, map_location='cpu')
    state = state.get('model', state)
    prefix = net.pretrained_cfg['classifier'] + '.'
    missing, unexpected = net.load_state_dict({k: v for k, v in state.items() if not k.startswith(prefix)},
                                              strict=False)
    if unexpected or any(not k.startswith(prefix) for k in missing):
        raise RuntimeError(f'checkpoint does not match the backbone: missing {missing}, unexpected {unexpected}')
    return missing


def add_input_channels(net, extra=1):
    """Widen the first convolution; the new channels start at zero, so the initial encoder output is unchanged."""
    *parents, name = net.pretrained_cfg['first_conv'].split('.')
    owner = net
    for p in parents:
        owner = getattr(owner, p)
    old = getattr(owner, name)
    new = nn.Conv2d(old.in_channels + extra, old.out_channels, old.kernel_size, old.stride, old.padding, old.dilation,
                    old.groups, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(owner, name, new)
    return net


def freeze_backbone(net):
    """Train only the classifier (decoupled retraining, Kang et al., 2020)."""
    for p in net.parameters():
        p.requires_grad = False
    for p in net.get_classifier().parameters():
        p.requires_grad = True
