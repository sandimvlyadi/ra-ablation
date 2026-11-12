"""RQ1-18: SKELEX (M015; Hugging Face skhoha/SKELEX, ViTMAEForPreTraining, ViT-L/16, 224 px) for timm, addendum 7.

  convert  map the encoder of the Hugging Face checkpoint to timm vit_large_patch16_224 (1 input channel) and write
           encoder.pt ({'model': state dict}) for train.py --init-from. The MAE decoder is dropped. SKELEX expects
           a grayscale radiograph copied to 3 channels and normalised with the ImageNet mean/std; both are folded into
           the patch embedding, so the 1-channel [0, 1] input of this pipeline gives the same embedding:
             W1 = sum_c W_c / s_c,   b1 = b - sum_c sum_hw W_c m_c / s_c.
  verify   compare the converted timm model with an independent forward pass written from the Hugging Face weights
           (pre-LN ViT, CLS token after the final LayerNorm, as ViTForImageClassification uses).

Usage: python -m ramscore.skelex convert --src ../data/external/skelex --out ../data/external/skelex/encoder_timm.pt
       python -m ramscore.skelex verify --src ../data/external/skelex --out ../data/external/skelex/encoder_timm.pt
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

TIMM_NAME = 'vit_large_patch16_224'


def hf_to_timm(hf, mean, std, depth):
    """Encoder weights of a Hugging Face ViT(-MAE) state dict, renamed to timm; normalisation folded in."""
    p = 'vit.'
    w = hf[p + 'embeddings.patch_embeddings.projection.weight'].float()
    b = hf[p + 'embeddings.patch_embeddings.projection.bias'].float()
    m = torch.tensor(mean, dtype=torch.float32)[None, :, None, None]
    s = torch.tensor(std, dtype=torch.float32)[None, :, None, None]
    out = {
        'cls_token': hf[p + 'embeddings.cls_token'],
        'pos_embed': hf[p + 'embeddings.position_embeddings'],
        'patch_embed.proj.weight': (w / s).sum(dim=1, keepdim=True),
        'patch_embed.proj.bias': b - (w * m / s).sum(dim=(1, 2, 3)),
        'norm.weight': hf[p + 'layernorm.weight'], 'norm.bias': hf[p + 'layernorm.bias'],
    }
    pairs = {'attn.proj': 'attention.output.dense', 'norm1': 'layernorm_before', 'norm2': 'layernorm_after',
             'mlp.fc1': 'intermediate.dense', 'mlp.fc2': 'output.dense'}
    for i in range(depth):
        h, t = f'{p}encoder.layer.{i}.', f'blocks.{i}.'
        a = h + 'attention.attention.'
        out[t + 'attn.qkv.weight'] = torch.cat([hf[a + f'{x}.weight'] for x in ('query', 'key', 'value')])
        out[t + 'attn.qkv.bias'] = torch.cat([hf[a + f'{x}.bias'] for x in ('query', 'key', 'value')])
        for tk, hk in pairs.items():
            out[t + tk + '.weight'] = hf[h + hk + '.weight']
            out[t + tk + '.bias'] = hf[h + hk + '.bias']
    return {k: v.float().contiguous() for k, v in out.items()}


def reference_cls(hf, x3, heads, depth, patch=16, eps=1e-12):
    """Independent pre-LN ViT forward from the Hugging Face weights on a normalised 3-channel batch: final CLS."""
    p = 'vit.'

    def g(k):
        return hf[k].float()

    x = F.conv2d(x3, g(p + 'embeddings.patch_embeddings.projection.weight'),
                 g(p + 'embeddings.patch_embeddings.projection.bias'), stride=patch).flatten(2).transpose(1, 2)
    x = torch.cat([g(p + 'embeddings.cls_token').expand(len(x), -1, -1), x], 1) + g(p + 'embeddings.position_embeddings')
    d = x.shape[-1]
    for i in range(depth):
        h = f'{p}encoder.layer.{i}.'

        def lin(z, k):
            return F.linear(z, g(h + k + '.weight'), g(h + k + '.bias'))

        y = F.layer_norm(x, (d,), g(h + 'layernorm_before.weight'), g(h + 'layernorm_before.bias'), eps)
        q, k, v = (lin(y, f'attention.attention.{n}').view(len(x), -1, heads, d // heads).transpose(1, 2)
                   for n in ('query', 'key', 'value'))
        att = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(len(x), -1, d)
        x = x + lin(att, 'attention.output.dense')
        y = F.layer_norm(x, (d,), g(h + 'layernorm_after.weight'), g(h + 'layernorm_after.bias'), eps)
        x = x + lin(F.gelu(lin(y, 'intermediate.dense')), 'output.dense')
    x = F.layer_norm(x, (d,), g(p + 'layernorm.weight'), g(p + 'layernorm.bias'), eps)
    return x[:, 0]


def load_source(src):
    from safetensors.torch import load_file

    src = Path(src)
    cfg = json.loads((src / 'config.json').read_text())
    pre = json.loads((src / 'preprocessor_config.json').read_text())
    hf = {k: v for k, v in load_file(str(src / 'model.safetensors')).items() if k.startswith('vit.')}
    return hf, cfg, pre


def timm_model(state, name=TIMM_NAME, **kwargs):
    import timm

    net = timm.create_model(name, pretrained=False, in_chans=1, **kwargs)
    missing, unexpected = net.load_state_dict(state, strict=False)
    head = net.pretrained_cfg['classifier'] + '.'
    if unexpected or any(not k.startswith(head) for k in missing):
        raise RuntimeError(f'conversion does not match {name}: missing {missing}, unexpected {unexpected}')
    return net


def convert(src, out):
    hf, cfg, pre = load_source(src)
    assert (cfg['hidden_size'], cfg['num_hidden_layers'], cfg['patch_size'], cfg['image_size']) == (1024, 24, 16, 224)
    state = hf_to_timm(hf, pre['image_mean'], pre['image_std'], cfg['num_hidden_layers'])
    timm_model(state)                         # every encoder key must land
    torch.save({'model': state, 'source': 'skhoha/SKELEX', 'timm': TIMM_NAME}, out)
    digest = hashlib.sha256(Path(out).read_bytes()).hexdigest()
    Path(str(out) + '.sha256').write_text(f'{digest}  {Path(out).name}\n')
    print(f'{len(state)} tensors -> {out} (sha256 {digest[:12]}...)')


def compare(net, hf, mean, std, heads, depth, size=224, n=2, seed=0):
    """Minimum cosine and maximum relative error between the timm CLS feature and the reference."""
    x = torch.rand(n, 1, size, size, generator=torch.Generator().manual_seed(seed))
    m = torch.tensor(mean)[None, :, None, None]
    s = torch.tensor(std)[None, :, None, None]
    with torch.no_grad():
        ref = reference_cls(hf, (x.expand(-1, 3, -1, -1) - m) / s, heads, depth)
        got = net.eval()(x)
    return F.cosine_similarity(ref, got).min().item(), ((ref - got).abs().max() / ref.abs().max()).item()


def verify(src, out):
    hf, cfg, pre = load_source(src)
    net = timm_model(torch.load(out, map_location='cpu')['model'], num_classes=0)
    cos, err = compare(net, hf, pre['image_mean'], pre['image_std'], cfg['num_attention_heads'],
                       cfg['num_hidden_layers'])
    print(f'min cosine {cos:.6f}, max relative error {err:.2e} (timm LayerNorm eps 1e-6 vs 1e-12 in SKELEX)')
    if cos < 0.9999:
        raise SystemExit('conversion check failed')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['convert', 'verify'])
    p.add_argument('--src', required=True, help='folder with config.json, preprocessor_config.json, model.safetensors')
    p.add_argument('--out', required=True, help='timm encoder.pt')
    args = p.parse_args(argv)
    (convert if args.command == 'convert' else verify)(args.src, args.out)


if __name__ == '__main__':
    main()
