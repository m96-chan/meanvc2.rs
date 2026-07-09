#!/usr/bin/env python3
"""Convert official CosyVoice2-0.5B VC-path checkpoints to fp32 safetensors.

Produces (under --out, default ./ckpt):
  cosyvoice_tokenizer.safetensors  - FSQ speech tokenizer v2 (S3Tokenizer naming)
  cosyvoice_campplus.safetensors   - CAM++ speaker encoder (3D-Speaker naming)
  cosyvoice_flow.safetensors       - UpsampleConformer + causal CFM U-Net
                                     (+ the seed-0 fixed CFM noise as `rand_noise`)
  cosyvoice_hift.safetensors       - HiFT vocoder, weight norm folded
  cosyvoice_mel.safetensors        - mel filterbanks: whisper 128@16k, hifigan 80@24k

Sources:
  --cosyvoice-dir: a downloaded FunAudioLLM/CosyVoice2-0.5B snapshot
                   (flow.pt, hift.pt, speech_tokenizer_v2.onnx). Apache-2.0.
  CAM++ weights: campplus.onnx in the snapshot is BN-fused (original names lost),
  so we convert from the numerically identical `campplus_cn_common.bin`
  (3D-Speaker / ModelScope iic/speech_campplus_sv_zh-cn_16k-common, Apache-2.0),
  either via --campplus-bin or an automatic modelscope download.

Deps: torch, safetensors, s3tokenizer, librosa, openai-whisper, modelscope.
"""
import argparse
import os

import numpy as np
import torch
from safetensors.torch import save_file


def fp32(sd: dict) -> dict:
    return {k: v.float().contiguous() for k, v in sd.items() if v.dtype.is_floating_point}


def fold_weight_norm(sd: dict) -> dict:
    """Fold torch parametrized weight norm (original0=g, original1=v) into plain weights."""
    out = {}
    for k, v in sd.items():
        if k.endswith('.parametrizations.weight.original0'):
            base = k[: -len('.parametrizations.weight.original0')]
            g = sd[k]
            w = sd[base + '.parametrizations.weight.original1']
            norm = w.norm(dim=list(range(1, w.dim())), keepdim=True)
            out[base + '.weight'] = (g * w / norm).contiguous()
        elif k.endswith('.parametrizations.weight.original1'):
            continue
        else:
            out[k] = v
    return out


def convert_flow(model_dir: str, out_dir: str):
    sd = torch.load(os.path.join(model_dir, 'flow.pt'), map_location='cpu', weights_only=True)
    sd = fp32(sd)
    # CausalConditionalCFM fixes its ODE start noise at construction with seed 0
    # (cosyvoice/flow/flow_matching.py) - export it so the Rust port is bit-identical.
    torch.manual_seed(0)
    sd['rand_noise'] = torch.randn([1, 80, 50 * 300])
    save_file(sd, os.path.join(out_dir, 'cosyvoice_flow.safetensors'))
    print(f'flow: {len(sd)} tensors')


def convert_hift(model_dir: str, out_dir: str):
    sd = torch.load(os.path.join(model_dir, 'hift.pt'), map_location='cpu', weights_only=True)
    sd = {k.replace('generator.', ''): v for k, v in sd.items()}
    sd = fold_weight_norm(fp32(sd))
    save_file(sd, os.path.join(out_dir, 'cosyvoice_hift.safetensors'))
    print(f'hift: {len(sd)} tensors')


def convert_tokenizer(model_dir: str, out_dir: str):
    from s3tokenizer.utils import onnx2torch
    sd = onnx2torch(os.path.join(model_dir, 'speech_tokenizer_v2.onnx'))
    sd = fp32(sd)
    save_file(sd, os.path.join(out_dir, 'cosyvoice_tokenizer.safetensors'))
    print(f'tokenizer: {len(sd)} tensors')


def convert_campplus(campplus_bin: str, out_dir: str):
    if campplus_bin is None:
        from modelscope import snapshot_download
        d = snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common')
        campplus_bin = os.path.join(d, 'campplus_cn_common.bin')
    sd = torch.load(campplus_bin, map_location='cpu', weights_only=True)
    sd = fp32(sd)
    save_file(sd, os.path.join(out_dir, 'cosyvoice_campplus.safetensors'))
    print(f'campplus: {len(sd)} tensors (from {campplus_bin})')


def convert_mel(out_dir: str):
    import whisper
    from librosa.filters import mel as librosa_mel
    # whisper log-mel 128 @16k (tokenizer front-end): n_fft 400, hop 160
    wf = whisper.audio.mel_filters(torch.device('cpu'), 128)  # (128, 201)
    # hifigan/matcha mel 80 @24k (prompt feats + vocoder target): n_fft 1920, hop 480, fmax 8000
    hf = torch.from_numpy(librosa_mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000)).float()
    save_file({'whisper_mel_fb_128_16k': wf.contiguous(), 'mel_fb_80_24k': hf.contiguous()},
              os.path.join(out_dir, 'cosyvoice_mel.safetensors'))
    print(f'mel: whisper {tuple(wf.shape)}, hifigan {tuple(hf.shape)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cosyvoice-dir', required=True,
                    help='FunAudioLLM/CosyVoice2-0.5B snapshot dir')
    ap.add_argument('--campplus-bin', default=None,
                    help='campplus_cn_common.bin path (default: modelscope download)')
    ap.add_argument('--out', default='ckpt')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    convert_flow(args.cosyvoice_dir, args.out)
    convert_hift(args.cosyvoice_dir, args.out)
    convert_tokenizer(args.cosyvoice_dir, args.out)
    convert_campplus(args.campplus_bin, args.out)
    convert_mel(args.out)


if __name__ == '__main__':
    main()
