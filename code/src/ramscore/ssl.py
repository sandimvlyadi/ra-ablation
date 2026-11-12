"""Self-supervised objectives for joint ROIs (proposal A3-A5).

- MAE (He et al., 2022): ViT encoder sees 25% of the patches; a small
  transformer decoder reconstructs the normalised pixels of the masked ones.
- DINO (Caron et al., 2021): student/teacher self-distillation with multi-crop,
  centring and sharpening; works with any timm backbone. The core of DINOv2
  (Oquab et al., 2023) without its iBOT and KoLeo terms.
- SimCLR (Chen et al., 2020): NT-Xent between two views, the other 2N - 2 views of the batch as negatives.
- BYOL (Grill et al., 2020): online projector + predictor regress the EMA target's projection of the other view.
- ChronoCon (Watzenboeck et al., 2026): Rank-N-Contrast-style loss where the
  negatives of a pair are the later (forward) or earlier (backward) visits of the
  same patient, side and ROI. Similarity is -L2 / tau with tau = 1, as in the
  paper; it is added to MAE or DINO with a weight (A5).

Every model exposes `.backbone`, a timm model with num_classes=0 whose state dict
loads into ramscore.models.build_model via load_backbone.
"""
import copy
import math

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch import nn


def make_backbone(name, dynamic_img_size=False):
    kwargs = {'dynamic_img_size': True} if dynamic_img_size and 'vit' in name else {}
    return timm.create_model(name, num_classes=0, in_chans=1, **kwargs)


# ---------------------------------------------------------------- augmentation

def _to_tensor(img):
    return torch.from_numpy(img.astype(np.float32) / 255.0)[None]


def _crop(size, scale, blur, solarize=0.0):
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=scale),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(0.4, 0.4, p=0.8),
        A.GaussianBlur(sigma_limit=(0.1, 2.0), p=blur),
        A.Solarize(threshold_range=(0.5, 0.5), p=solarize),
    ])


class MultiCrop:
    """DINO views: 2 global crops (224) then `n_local` local crops (96)."""

    def __init__(self, n_local=6, global_size=224, local_size=96):
        self.global1 = _crop(global_size, (0.4, 1.0), blur=1.0)
        self.global2 = _crop(global_size, (0.4, 1.0), blur=0.1, solarize=0.2)
        self.local = _crop(local_size, (0.1, 0.4), blur=0.5)
        self.n_local = n_local

    def __call__(self, img):
        views = [self.global1(image=img)['image'], self.global2(image=img)['image']]
        views += [self.local(image=img)['image'] for _ in range(self.n_local)]
        return [_to_tensor(v) for v in views]


class Crops:
    """`n` independent crops of one size (MAE: n = 1; ChronoCon double-crop: n = 2)."""

    def __init__(self, n=1, size=224, scale=(0.4, 1.0)):
        self.n = n
        self.t = A.Compose([A.RandomResizedCrop(size=(size, size), scale=scale), A.HorizontalFlip(),
                            A.RandomBrightnessContrast(0.2, 0.2, p=0.5)])

    def __call__(self, img):
        return [_to_tensor(self.t(image=img)['image']) for _ in range(self.n)]


# ---------------------------------------------------------------- MAE

class MAE(nn.Module):
    def __init__(self, name='vit_small_patch16_224', mask_ratio=0.75, dec_dim=256, dec_depth=4, dec_heads=8):
        super().__init__()
        self.backbone = enc = make_backbone(name)
        if not hasattr(enc, 'patch_embed'):
            raise ValueError(f'MAE needs a ViT backbone, got {name}')
        self.mask_ratio = mask_ratio
        self.patch = enc.patch_embed.patch_size[0]
        n = enc.patch_embed.num_patches
        self.decoder_embed = nn.Linear(enc.embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, n + 1, dec_dim))
        self.decoder = nn.Sequential(*[Block(dec_dim, dec_heads, mlp_ratio=4, qkv_bias=True) for _ in range(dec_depth)])
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, self.patch ** 2)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)

    def patchify(self, img):
        p = self.patch
        b, c, h, w = img.shape
        x = img.reshape(b, c, h // p, p, w // p, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // p) * (w // p), p * p * c)

    def encode(self, img):
        enc = self.backbone
        x = enc.patch_embed(img) + enc.pos_embed[:, 1:]
        b, n, d = x.shape
        n_keep = int(n * (1 - self.mask_ratio))
        ids_shuffle = torch.rand(b, n, device=img.device).argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        x = torch.gather(x, 1, ids_shuffle[:, :n_keep, None].expand(-1, -1, d))
        mask = torch.ones(b, n, device=img.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)                  # 1 = masked, in original patch order
        cls = (enc.cls_token + enc.pos_embed[:, :1]).expand(b, -1, -1)
        x = enc.norm(enc.blocks(torch.cat([cls, x], dim=1)))
        return x, mask, ids_restore

    def forward(self, img):
        """Mean squared error on the per-patch normalised pixels of the masked patches."""
        x, mask, ids_restore = self.encode(img)
        x = self.decoder_embed(x)
        b, n = ids_restore.shape
        tokens = torch.cat([x[:, 1:], self.mask_token.expand(b, n + 1 - x.shape[1], -1)], dim=1)
        tokens = torch.gather(tokens, 1, ids_restore[..., None].expand(-1, -1, x.shape[2]))
        x = self.decoder_norm(self.decoder(torch.cat([x[:, :1], tokens], dim=1) + self.decoder_pos))
        pred = self.decoder_pred(x[:, 1:])
        target = self.patchify(img)
        target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
        return (((pred - target) ** 2).mean(-1) * mask).sum() / mask.sum()


# ---------------------------------------------------------------- DINO

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.Parameter(torch.empty(out_dim, bottleneck))  # weight-normalised, gain fixed at 1
        nn.init.trunc_normal_(self.last, std=0.02)

    def forward(self, x):
        return F.linear(F.normalize(self.mlp(x), dim=-1), F.normalize(self.last, dim=-1))


class DINO(nn.Module):
    def __init__(self, name='resnet34', out_dim=4096, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.backbone = make_backbone(name, dynamic_img_size=True)    # student
        self.head = DINOHead(self.backbone.num_features, out_dim)
        # The teacher stays in train mode like the student, so its BatchNorm layers keep
        # their own running statistics, as in the official DINO code.
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = copy.deepcopy(self.head)
        for p in self.teacher_parameters():
            p.requires_grad = False
        self.student_temp, self.center_momentum = student_temp, center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))

    def student_parameters(self):
        return [*self.backbone.parameters(), *self.head.parameters()]

    def teacher_parameters(self):
        return [*self.teacher_backbone.parameters(), *self.teacher_head.parameters()]

    def forward(self, views, teacher_temp):
        """Cross-entropy between teacher outputs on the 2 global views and student outputs on all other views."""
        with torch.no_grad():
            t_out = self.teacher_head(self.teacher_backbone(torch.cat(views[:2])))
            t = F.softmax((t_out - self.center) / teacher_temp, dim=-1).chunk(2)
        # global and local crops have different sizes, so run them as two batches
        s_out = [self.head(self.backbone(torch.cat(views[:2])))]
        if len(views) > 2:
            s_out.append(self.head(self.backbone(torch.cat(views[2:]))))
        s = F.log_softmax(torch.cat(s_out) / self.student_temp, dim=-1).chunk(len(views))
        loss, n = 0.0, 0
        for i, ti in enumerate(t):
            for j, sj in enumerate(s):
                if i != j:
                    loss = loss + (-ti * sj).sum(-1).mean()
                    n += 1
        self.update_center(t_out)
        return loss / n

    @torch.no_grad()
    def update_center(self, t_out):
        self.center.mul_(self.center_momentum).add_(t_out.float().mean(0, keepdim=True), alpha=1 - self.center_momentum)

    @torch.no_grad()
    def update_teacher(self, momentum):
        for s, t in zip(self.student_parameters(), self.teacher_parameters()):
            t.mul_(momentum).add_(s.detach(), alpha=1 - momentum)


# ---------------------------------------------------------------- SimCLR and BYOL

def mlp(in_dim, hidden, out_dim):
    """Two-layer projector / predictor with BatchNorm (SimCLR and BYOL official code)."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, out_dim))


def nt_xent(z, temperature=0.1):
    """NT-Xent of SimCLR on 2N embeddings ordered [view 1 of 0..N-1, view 2 of 0..N-1]: each view's positive is the
    other view of the same image, every other view in the batch is a negative."""
    z = F.normalize(z.float(), dim=1)
    n = z.shape[0] // 2
    sim = (z @ z.T / temperature).masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float('-inf'))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, target)


class SimCLR(nn.Module):
    def __init__(self, name='resnet34', proj_dim=128, temperature=0.1):
        super().__init__()
        self.backbone = make_backbone(name)
        nf = self.backbone.num_features
        self.projector = mlp(nf, nf, proj_dim)
        self.temperature = temperature

    def forward(self, views):
        return nt_xent(self.projector(self.backbone(torch.cat(views[:2]))), self.temperature)


def byol_regression(p, z):
    """2 - 2 cos(p, z) per sample, averaged (Grill et al., 2020, eq. 2); z is the target, no gradient."""
    return (2 - 2 * F.cosine_similarity(p.float(), z.detach().float(), dim=-1)).mean()


class BYOL(nn.Module):
    def __init__(self, name='resnet34', hidden=4096, proj_dim=256):
        super().__init__()
        self.backbone = make_backbone(name)                          # online encoder: the one kept for fine-tuning
        nf = self.backbone.num_features
        self.projector = mlp(nf, hidden, proj_dim)
        self.predictor = mlp(proj_dim, hidden, proj_dim)
        # target stays in train mode like the online network (BatchNorm statistics of its own, as in DINO above)
        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.teacher_parameters():
            p.requires_grad = False

    def student_parameters(self):
        return [*self.backbone.parameters(), *self.projector.parameters(), *self.predictor.parameters()]

    def teacher_parameters(self):
        return [*self.target_backbone.parameters(), *self.target_projector.parameters()]

    def forward(self, views):
        """Symmetric loss: the prediction from each view regresses the target projection of the other view."""
        x = torch.cat(views[:2])
        p1, p2 = self.predictor(self.projector(self.backbone(x))).chunk(2)
        with torch.no_grad():
            z1, z2 = self.target_projector(self.target_backbone(x)).chunk(2)
        return byol_regression(p1, z2) + byol_regression(p2, z1)

    @torch.no_grad()
    def update_teacher(self, momentum):
        for s, t in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            t.mul_(momentum).add_(s.detach(), alpha=1 - momentum)
        for s, t in zip(self.projector.parameters(), self.target_projector.parameters()):
            t.mul_(momentum).add_(s.detach(), alpha=1 - momentum)


# ---------------------------------------------------------------- ChronoCon

def _directional_loss(sim, same, pair, beyond):
    """Mean RnC term over valid (anchor, positive) pairs.

    sim[a, n]: similarity; same[a, n]: same series and n != a; pair[a, p]: allowed pairs;
    beyond[p, n]: n lies past p in this direction, so it is a negative for (a, p).
    """
    negatives = pair[:, :, None] & same[:, None, :] & beyond[None, :, :]   # [a, p, n]
    valid = negatives.any(-1)
    if not valid.any():
        return sim.new_zeros(())
    neg_sim = sim[:, None, :].expand_as(negatives).masked_fill(~negatives, float('-inf'))
    log_denominator = torch.logsumexp(torch.cat([sim[:, :, None], neg_sim], dim=-1), dim=-1)
    return (log_denominator - sim)[valid].mean()


def chronocon_loss(features, groups, times, tau=1.0):
    """ChronoCon loss (Watzenboeck et al., 2026, eq. 2-3) for one batch of whole series.

    features: [N, d]; groups: [N] series id; times: [N] visit time. Two crops of the
    same image share group and time, which lets series of two visits contribute.
    """
    x = features.float()
    sq = (x * x).sum(-1)
    # clamped so the gradient of the distance stays finite for (near-)identical crops
    dist = (sq[:, None] + sq[None, :] - 2 * x @ x.T).clamp_min(1e-12).sqrt()
    sim = -dist / tau
    n = len(features)
    same = (groups[:, None] == groups[None, :]) & ~torch.eye(n, dtype=torch.bool, device=features.device)
    t_row, t_col = times[:, None], times[None, :]
    forward = _directional_loss(sim, same, same & (t_row <= t_col), t_row < t_col)    # t_a <= t_p < t_n
    backward = _directional_loss(sim, same, same & (t_row >= t_col), t_row > t_col)   # t_a >= t_p > t_n
    return forward + backward


# ---------------------------------------------------------------- schedules

def cosine_schedule(base, final, total_steps, warmup_steps=0, start=0.0):
    """Per-step values: linear warm-up from `start` to `base`, then cosine to `final`."""
    warm = np.linspace(start, base, warmup_steps, endpoint=False)
    steps = np.arange(total_steps - warmup_steps)
    cos = final + 0.5 * (base - final) * (1 + np.cos(math.pi * steps / max(len(steps), 1)))
    return np.concatenate([warm, cos])
