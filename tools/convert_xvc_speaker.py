#!/usr/bin/env python3
"""Converts the ERes2Net speaker encoder shipped inside the official X-VC
checkpoint (`ckpts/xvc.pt`, generator keys `speaker_encoder.model.*`) to
`ckpt/xvc_speaker.safetensors`, loadable by `xvc::speaker::SpeakerEncoder`.

The encoder originates from ModelScope
`iic/speech_eres2net_sv_en_voxceleb_16k`, but **`xvc.pt` is the ground
truth**: X-VC training ran the module in train mode, so every BatchNorm
running-stat buffer drifted from the ModelScope release (279 buffer tensors
differ; the learnable parameters are bit-identical). The script reports that
drift as a sanity check when the pretrained ckpt is present.

Tensor names are kept 1:1 with the official `ERes2Net` module tree
(`conv1.weight`, `layer1.0.convs.0.weight`, `fuse_mode12.local_att.0.weight`,
`seg_1.weight`, ...); only the integer `num_batches_tracked` BatchNorm
counters are dropped (inference uses the running stats). There are no
weight_norm parametrizations to fold (asserted).

Usage: python3 tools/convert_xvc_speaker.py [--xvc-repo PATH] [--ckpt ckpt]
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

PREFIX = "speaker_encoder.model."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xvc-repo",
        type=Path,
        default=Path.home() / "tmp/xvc-recon/X-VC",
        help="clone of Jerrister/X-VC with the released checkpoints",
    )
    parser.add_argument(
        "--ckpt", type=Path, default=Path("ckpt"), help="output directory"
    )
    args = parser.parse_args()

    gen = torch.load(
        args.xvc_repo / "ckpts/xvc.pt", map_location="cpu", weights_only=True, mmap=True
    )["generator"]
    state = {k[len(PREFIX) :]: v for k, v in gen.items() if k.startswith(PREFIX)}
    assert state, f"no {PREFIX}* keys in xvc.pt"
    assert not any("parametrizations" in k for k in state), "unexpected weight_norm"

    pretrained = (
        args.xvc_repo
        / "pretrained/speech_eres2net_sv_en_voxceleb_16k/pretrained_eres2net.ckpt"
    )
    if pretrained.exists():
        pre = torch.load(pretrained, map_location="cpu", weights_only=True)
        drift = [k for k in state if not torch.equal(state[k].float(), pre[k].float())]
        assert all("running_" in k or "num_batches_tracked" in k for k in drift)
        print(
            f"vs ModelScope ckpt: {len(drift)} BatchNorm buffers drifted during "
            "X-VC training, parameters identical"
        )

    sd = {
        k: v.contiguous().float()
        for k, v in state.items()
        if not k.endswith("num_batches_tracked")
    }
    out = args.ckpt / "xvc_speaker.safetensors"
    save_file(sd, str(out))
    n_params = sum(v.numel() for v in sd.values())
    print(f"{out}: {len(sd)} tensors ({len(state)} in xvc.pt), {n_params / 1e6:.2f}M params")


if __name__ == "__main__":
    main()
