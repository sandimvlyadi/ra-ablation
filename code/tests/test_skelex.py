"""SKELEX conversion (addendum 7): Hugging Face ViT-MAE encoder -> timm ViT, normalisation folded in."""
import torch
from timm.models.vision_transformer import VisionTransformer

from ramscore.skelex import compare, hf_to_timm

D, DEPTH, HEADS, MLP, PATCH, SIZE = 32, 2, 4, 64, 16, 32
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def fake_hf(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.2
    n = (SIZE // PATCH) ** 2 + 1
    hf = {'vit.embeddings.cls_token': r(1, 1, D), 'vit.embeddings.position_embeddings': r(1, n, D),
          'vit.embeddings.patch_embeddings.projection.weight': r(D, 3, PATCH, PATCH),
          'vit.embeddings.patch_embeddings.projection.bias': r(D),
          'vit.layernorm.weight': 1 + r(D), 'vit.layernorm.bias': r(D)}
    for i in range(DEPTH):
        h = f'vit.encoder.layer.{i}.'
        for x in ('query', 'key', 'value'):
            hf[h + f'attention.attention.{x}.weight'], hf[h + f'attention.attention.{x}.bias'] = r(D, D), r(D)
        hf[h + 'attention.output.dense.weight'], hf[h + 'attention.output.dense.bias'] = r(D, D), r(D)
        hf[h + 'intermediate.dense.weight'], hf[h + 'intermediate.dense.bias'] = r(MLP, D), r(MLP)
        hf[h + 'output.dense.weight'], hf[h + 'output.dense.bias'] = r(D, MLP), r(D)
        for ln in ('layernorm_before', 'layernorm_after'):
            hf[h + ln + '.weight'], hf[h + ln + '.bias'] = 1 + r(D), r(D)
    return hf


def tiny_vit():
    return VisionTransformer(img_size=SIZE, patch_size=PATCH, in_chans=1, num_classes=0, embed_dim=D, depth=DEPTH,
                             num_heads=HEADS, mlp_ratio=MLP / D, qkv_bias=True)


def test_converted_vit_matches_the_reference_forward():
    hf = fake_hf()
    net = tiny_vit()
    missing, unexpected = net.load_state_dict(hf_to_timm(hf, MEAN, STD, DEPTH), strict=False)
    assert not unexpected and not missing
    cos, err = compare(net, hf, MEAN, STD, HEADS, DEPTH, size=SIZE, n=3)
    assert cos > 0.99999 and err < 1e-3


def test_without_folding_the_features_differ():
    hf = fake_hf()
    net = tiny_vit()
    net.load_state_dict(hf_to_timm(hf, [0, 0, 0], [1, 1, 1], DEPTH), strict=False)   # normalisation ignored
    cos, _ = compare(net, hf, MEAN, STD, HEADS, DEPTH, size=SIZE, n=3)
    assert cos < 0.999


def test_grad_checkpointing_keeps_outputs_and_gradients():
    torch.manual_seed(0)
    a, b = tiny_vit(), tiny_vit()
    b.load_state_dict(a.state_dict())
    b.set_grad_checkpointing(True)
    x = torch.rand(2, 1, SIZE, SIZE)
    a.train(), b.train()
    ya, yb = a(x), b(x)
    ya.sum().backward(), yb.sum().backward()
    assert torch.allclose(ya, yb, atol=1e-6)
    assert torch.allclose(a.blocks[0].attn.qkv.weight.grad, b.blocks[0].attn.qkv.weight.grad, atol=1e-5)
