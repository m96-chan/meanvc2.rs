#!/usr/bin/env python3
"""Exports the X-VC semantic front-end weights to
`ckpt/xvc_tokenizer.safetensors` for the Rust port (issue #30):

* the frozen **GLM-4-Voice tokenizer** (`WhisperVQEncoder`: Whisper-large-v3
  encoder truncated to 16 layers with causal convs + block-causal attention,
  avg-pool k=4 -> 12.5 Hz, 16384x1280 VQ codebook, learned post-VQ
  positional embedding), read from the `zai-org/glm-4-voice-tokenizer`
  safetensors snapshot;
* the **semantic adapter** (`semantic_adapter.*`, Vocos-ConvNeXt
  `Decoder_with_upsample`, 12.5 -> 50 Hz, 1280 -> 1024 channels), read from
  the `generator` state dict of the official `ckpts/xvc.pt`.

Parameter names are kept 1:1 with the official modules: the encoder tensors
use the GLM safetensors names (`conv1.*`, `layers.{i}.self_attn.*`,
`codebook.weight`, `embed_positions.weight`, `embed_positions2.weight`, ...;
identical to `semantic_encoder.encoder.*` in `xvc.pt` minus the prefix — the
Phase-0 recon verified the two are bit-identical) and the adapter tensors
keep their `semantic_adapter.*` prefix.

The VQ EMA training buffers (`ema_count`, `ema_weight`) are dropped. Any
`weight_norm` parametrization pairs (`...parametrizations.weight.original0/1`)
would be folded into plain weights (g * v / ||v||, norm over all dims but 0)
— none of these modules actually use weight_norm, the fold is a guard.

**Dtype: everything is stored as float32** (the checkpoints already are
fp32; the script asserts it rather than silently casting).

Usage:

    python3 tools/convert_xvc_tokenizer.py [--xvc-repo ~/tmp/xvc-recon/X-VC] \
        [--out ckpt/xvc_tokenizer.safetensors]

Prerequisites: torch + safetensors, and the official X-VC clone with
`ckpts/xvc.pt` and a `glm-4-voice-tokenizer/` snapshot (see
tools/gen_xvc_fixtures.py for download pointers).
"""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent

# VQ EMA buffers: training-time only, not part of the inference graph.
DROP = {"ema_count", "ema_weight"}


def fold_weight_norm(state: dict) -> dict:
    """Folds torch `weight_norm` parametrizations into plain weights:
    `weight = g * v / ||v||` with the norm over every dim except 0
    (`parametrizations.weight.original0` = g, `original1` = v)."""
    out = {}
    for key, value in state.items():
        if key.endswith("parametrizations.weight.original0"):
            base = key[: -len("parametrizations.weight.original0")]
            g = value
            v = state[base + "parametrizations.weight.original1"]
            norm = v.norm(p=2, dim=tuple(range(1, v.dim())), keepdim=True)
            out[base + "weight"] = g * v / norm
        elif key.endswith("parametrizations.weight.original1"):
            continue
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xvc-repo", type=Path, default=Path("~/tmp/xvc-recon/X-VC"),
        help="official Jerrister/X-VC clone with checkpoints in place",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "ckpt/xvc_tokenizer.safetensors",
    )
    args = parser.parse_args()
    repo = args.xvc_repo.expanduser().resolve()

    tensors = {}

    # 1) GLM-4-Voice tokenizer (frozen Whisper-VQ encoder).
    tok_path = repo / "glm-4-voice-tokenizer/model.safetensors"
    with safe_open(str(tok_path), framework="pt") as f:
        for key in f.keys():
            if key in DROP:
                continue
            tensors[key] = f.get_tensor(key)

    # 2) Semantic adapter from the xvc.pt generator.
    ckpt = torch.load(repo / "ckpts/xvc.pt", map_location="cpu", mmap=True)
    generator = ckpt["generator"]
    adapter = {
        k: v for k, v in generator.items() if k.startswith("semantic_adapter.")
    }
    assert adapter, "no semantic_adapter.* tensors in ckpts/xvc.pt generator"
    tensors.update(fold_weight_norm(adapter))

    for key, value in tensors.items():
        assert "parametrizations" not in key, f"unfolded weight_norm: {key}"
        assert value.dtype == torch.float32, f"{key}: expected fp32, got {value.dtype}"
    tensors = {k: v.contiguous() for k, v in tensors.items()}

    n_enc = sum(v.numel() for k, v in tensors.items() if not k.startswith("semantic_adapter."))
    n_ada = sum(v.numel() for k, v in tensors.items() if k.startswith("semantic_adapter."))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    print(
        f"{args.out}: {len(tensors)} tensors, fp32 — "
        f"tokenizer {n_enc / 1e6:.1f}M + semantic_adapter {n_ada / 1e6:.1f}M "
        f"= {(n_enc + n_ada) / 1e6:.1f}M params "
        f"({(n_enc + n_ada) * 4 / 2**30:.2f} GiB)"
    )


if __name__ == "__main__":
    main()
