#!/usr/bin/env python3
"""Convert the official Seed-VC checkpoints to fp32 safetensors for
`crates/seedvc` (issue #50).

Target model: `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned`
(the issue-#49 A/B winner) plus its companions:

  - net.cfm.estimator (DiT + WaveNet refiner)  -> seedvc_dit.safetensors
  - net.length_regulator                        -> seedvc_regulator.safetensors
  - net.style_encoder (CAM++)                   -> seedvc_campplus.safetensors
    (falls back to the standalone funasr campplus_cn_common.bin)
  - nvidia/bigvgan_v2_22khz_80band_256x         -> seedvc_bigvgan.safetensors
    (weight norm folded like the official `remove_weight_norm()`)
  - openai/whisper-small encoder                -> seedvc_whisper.safetensors

Run inside the seed-vc environment (weights in its HF cache):

    python tools/convert_seedvc.py --seedvc-dir <seed-vc clone> --out ckpt/

Everything stays fp32 (project policy: no quantization).
"""
import argparse
import glob
import os

import torch
from safetensors.torch import save_file


def fp32(sd, prefix=""):
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        out[prefix + k] = v.detach().to(torch.float32).contiguous()
    return out


def fold_weight_norm(sd):
    """Fold `*_g` / `*_v` (or parametrizations) pairs into plain weights,
    mirroring `remove_weight_norm()`."""
    out = {}
    done = set()
    for k in sd:
        if k.endswith("weight_g"):
            base = k[: -len("weight_g")]
            g, v = sd[k], sd[base + "weight_v"]
            w = v * (g / v.norm(2, dim=list(range(1, v.dim())), keepdim=True))
            out[base + "weight"] = w.to(torch.float32).contiguous()
            done.add(k)
            done.add(base + "weight_v")
    for k, v in sd.items():
        if k in done or not torch.is_tensor(v):
            continue
        out[k] = v.to(torch.float32).contiguous()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seedvc-dir", required=True)
    ap.add_argument("--out", default="ckpt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def find(pattern):
        hits = glob.glob(
            os.path.join(args.seedvc_dir, "checkpoints", "**", pattern), recursive=True
        )
        assert hits, f"not found under checkpoints/: {pattern}"
        return hits[0]

    # --- main checkpoint --------------------------------------------------
    ck = torch.load(
        find("DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
        map_location="cpu",
        weights_only=False,
    )["net"]
    save_file(fp32(ck["cfm"]), os.path.join(args.out, "seedvc_dit.safetensors"))
    save_file(
        fp32(ck["length_regulator"]),
        os.path.join(args.out, "seedvc_regulator.safetensors"),
    )
    save_file(
        fp32(ck["style_encoder"]),
        os.path.join(args.out, "seedvc_campplus.safetensors"),
    )
    print("dit / regulator / campplus written")

    # --- BigVGAN (weight norm folded) ------------------------------------
    bv = torch.load(find("models--nvidia--bigvgan*/**/bigvgan_generator.pt"),
                    map_location="cpu", weights_only=False)
    bv_sd = bv.get("generator", bv)
    save_file(
        fold_weight_norm(bv_sd), os.path.join(args.out, "seedvc_bigvgan.safetensors")
    )
    print("bigvgan written (weight norm folded)")

    # --- Whisper-small encoder -------------------------------------------
    from safetensors.torch import load_file

    wh = load_file(find("models--openai--whisper-small/**/model.safetensors"))
    enc = {k: v.to(torch.float32) for k, v in wh.items() if k.startswith("model.encoder.")}
    save_file(enc, os.path.join(args.out, "seedvc_whisper.safetensors"))
    print(f"whisper encoder written ({len(enc)} tensors)")


if __name__ == "__main__":
    main()
