#!/usr/bin/env python3
"""Exports the X-VC generator weights needed by `crates/xvc` from the
official checkpoint (`ckpts/xvc.pt` of https://github.com/Jerrister/X-VC,
weights from https://huggingface.co/chenxie95/X-VC) to safetensors
(issue #30, Phase 1):

| output                        | modules (official names kept 1:1)                |
|-------------------------------|---------------------------------------------------|
| `ckpt/xvc_codec.safetensors`      | `acoustic_encoder.*`, `acoustic_quantizer.*`, `acoustic_decoder.*` (the SAC acoustic codec) |
| `ckpt/xvc_converter.safetensors`  | `acoustic_converter.*` (the 6-block MMDiT converter) |

Weight-norm folding: the checkpoint stores every `WNConv1d` /
`WNConvTranspose1d` weight as a `torch.nn.utils.parametrizations.weight_norm`
pair `X.parametrizations.weight.original0` (the magnitude `g`) and
`...original1` (the direction `v`). Those are folded into a plain
`X.weight = g * v / ||v||` — exactly `torch._weight_norm(v, g, dim=0)`,
i.e. the norm is taken over every dim but 0, which is what
`remove_weight_norm()` computes at inference — so the Rust side loads
ordinary `Conv1d`/`ConvTranspose1d` weights.

Also dropped (training-only): `acoustic_quantizer.cluster_size` (the EMA
code-usage buffer of the FVQ dead-code expiry). `ema_generator` and
`discriminator` are never read. Everything else is saved as fp32.

Usage:

    python3 tools/convert_xvc_generator.py [--ckpt ckpt] [--xvc-repo ~/tmp/xvc-recon/X-VC]
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

CODEC_PREFIXES = ("acoustic_encoder.", "acoustic_quantizer.", "acoustic_decoder.")
CONVERTER_PREFIXES = ("acoustic_converter.",)
DROP_KEYS = {"acoustic_quantizer.cluster_size"}

WN_G_SUFFIX = ".parametrizations.weight.original0"
WN_V_SUFFIX = ".parametrizations.weight.original1"


def fold_weight_norm(sd: dict) -> dict:
    """Folds every weight_norm (g, v) pair into a plain `.weight` tensor."""
    out, folded = {}, 0
    for key, value in sd.items():
        if key.endswith(WN_G_SUFFIX):
            v = sd[key.removesuffix(WN_G_SUFFIX) + WN_V_SUFFIX]
            # remove_weight_norm(): w = g * v / ||v||, norm over dims != 0.
            out[key.removesuffix(WN_G_SUFFIX) + ".weight"] = torch._weight_norm(
                v, value, 0
            )
            folded += 1
        elif not key.endswith(WN_V_SUFFIX):
            out[key] = value
    print(f"  folded {folded} weight_norm pairs")
    return out


def export(sd: dict, prefixes: tuple, path: Path):
    sub = {
        k: v
        for k, v in sd.items()
        if k.startswith(prefixes) and k not in DROP_KEYS
    }
    sub = fold_weight_norm(sub)
    sub = {k: v.contiguous().float() for k, v in sub.items()}
    save_file(sub, str(path))
    params = sum(v.numel() for v in sub.values())
    print(f"{path}: {len(sub)} tensors, {params / 1e6:.1f} M params")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt"))
    parser.add_argument(
        "--xvc-repo", type=Path, default=Path("~/tmp/xvc-recon/X-VC").expanduser()
    )
    args = parser.parse_args()

    ckpt_path = args.xvc_repo / "ckpts" / "xvc.pt"
    print(f"loading {ckpt_path} (generator only) ...")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["generator"]

    export(sd, CODEC_PREFIXES, args.ckpt / "xvc_codec.safetensors")
    export(sd, CONVERTER_PREFIXES, args.ckpt / "xvc_converter.safetensors")


if __name__ == "__main__":
    main()
