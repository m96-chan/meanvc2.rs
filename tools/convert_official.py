#!/usr/bin/env python3
"""Converts the official ASLP-lab/MeanVC TorchScript checkpoints
(vocos.pt, fastu2++.pt) to safetensors loadable by this crate.
model_200ms.safetensors needs no conversion (MeanVc1::load reads it).

Usage: python3 tools/convert_official.py [ckpt_dir]
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ckpt = Path(sys.argv[1] if len(sys.argv) > 1 else "ckpt")

# Vocos: names already match backends::Vocos; drop jit buffers.
sd = torch.jit.load(ckpt / "vocos.pt", map_location="cpu").state_dict()
sd = {k: v.contiguous().float() for k, v in sd.items() if "istft.window" not in k}
save_file(sd, str(ckpt / "vocos.safetensors"))
print(f"vocos.safetensors: {len(sd)} tensors")

# Fast-U2++: strip the leading "encoder." to match backends::FastU2pp roots.
sd = torch.jit.load(ckpt / "fastu2pp.pt", map_location="cpu").state_dict()
sd = {k.removeprefix("encoder."): v.contiguous().float() for k, v in sd.items()}
save_file(sd, str(ckpt / "fastu2pp.safetensors"))
print(f"fastu2pp.safetensors: {len(sd)} tensors")
