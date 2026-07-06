#!/usr/bin/env python3
"""Regenerates every MeanVC v1 parity fixture for `tests/golden.rs`
(issue #14) deterministically into `ckpt/`.

Fixtures written (all consumed by the `v1_*` golden tests):

| file                            | contents                                        |
|---------------------------------|-------------------------------------------------|
| `dit_fixture.safetensors`       | single DiT forward `u(x, r=0, t=1)` (seed 7)    |
| `dit_stream_fixture.safetensors`| 8-chunk KV-cache streaming refs (seed 11)       |
| `copysyn_fixture.safetensors`   | official mel of test.wav + Vocos copy-synthesis |
| `asr_chunk0_fixture.safetensors`| Fast-U2++ embed / layer0 / chunk-0 references   |
| `pipeline_ref.safetensors`      | kaldi fbank + chunked-ASR BNF references        |

Prerequisites:

* the official checkpoints in `ckpt/`: `model_200ms.safetensors`,
  `config.json`, `vocos.pt`, `fastu2pp.pt`, `test.wav`
  (see https://huggingface.co/ASLP-lab/MeanVC);
* torch, safetensors, soundfile, torchaudio, librosa, einops,
  x_transformers (the DiT reference imports the official repo, which
  needs the last two);
* a clone of https://github.com/ASLP-lab/MeanVC for the reference
  `src/infer/dit_kvcache.py` DiT. Pass it with `--meanvc-repo`; without
  the flag the script shallow-clones into a temporary directory. The
  clone is patched in place (idempotently) so the module imports
  standalone: `src/model/__init__.py` is blanked and dit_kvcache's
  `from modules import ...` becomes `from src.infer.modules import ...`.

Usage:

    python3 tools/gen_v1_fixtures.py [--ckpt ckpt] [--meanvc-repo PATH]
    cargo test --release --test golden
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
from librosa.filters import mel as librosa_mel_fn
from safetensors.torch import load_file, save_file

MEANVC_URL = "https://github.com/ASLP-lab/MeanVC"

# Official DiT config for the released 200 ms checkpoint
# (`ckpt/config.json`, mel_dim added).
DIT_KWARGS = dict(
    dim=512, depth=4, heads=2, ff_mult=2, mel_dim=80, bn_dim=256,
    qk_norm="rms_norm", chunk_size=20, dropout=0.0,
)
CHUNK = 20
KV_CACHE_MAX_LEN = 100  # C_KV_CACHE_MAX_LEN in infer_ref.py


def load_test_wav(ckpt: Path) -> torch.Tensor:
    """test.wav as float32 in [-1, 1] (int16 / 32768, matching hound)."""
    wav, sr = sf.read(ckpt / "test.wav", dtype="int16")
    assert sr == 16_000, f"expected 16 kHz test.wav, got {sr}"
    return torch.from_numpy(wav).float() / 32768.0


def import_official_dit(repo: Path | None):
    """Imports the reference DiT from an (auto-cloned) MeanVC checkout,
    patching it for standalone use."""
    if repo is None:
        repo = Path(tempfile.mkdtemp(prefix="meanvc-official-")) / "MeanVC"
        print(f"cloning {MEANVC_URL} -> {repo}")
        subprocess.run(
            ["git", "clone", "--depth", "1", MEANVC_URL, str(repo)],
            check=True,
        )
    repo = repo.resolve()
    # Patch (idempotent): blank the package __init__ that drags in the
    # trainer, and make dit_kvcache's sibling import absolute.
    (repo / "src/model/__init__.py").write_text("")
    kvcache = repo / "src/infer/dit_kvcache.py"
    kvcache.write_text(
        kvcache.read_text().replace(
            "from modules import", "from src.infer.modules import"
        )
    )
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src/infer"))
    from src.infer.dit_kvcache import DiT  # noqa: PLC0415

    return DiT


def load_official_dit(DiT, ckpt: Path):
    model = DiT(**DIT_KWARGS)
    missing, unexpected = model.load_state_dict(
        load_file(ckpt / "model_200ms.safetensors"), strict=False
    )
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval()


@torch.no_grad()
def gen_dit_fixture(model, ckpt: Path):
    """Single non-streaming forward u(x, r=0, t=1), no cache (seed 7)."""
    torch.manual_seed(7)
    x = torch.randn(1, CHUNK, 80)
    bn = torch.randn(1, CHUNK, 256)
    prompts = torch.rand(1, 120, 80) * 2 - 1  # reference-mel domain [-1, 1]
    spks = torch.randn(1, 256)
    u, _ = model(
        x, torch.ones(1), torch.zeros(1), cache=None, cond=bn, spks=spks,
        prompts=prompts, is_inference=False,
    )
    save_file(
        {"x": x, "bn": bn, "prompts": prompts, "spks": spks, "u_ref": u},
        str(ckpt / "dit_fixture.safetensors"),
    )
    print("dit_fixture.safetensors: u_ref", tuple(u.shape))


@torch.no_grad()
def gen_dit_stream_fixture(model, ckpt: Path):
    """8-chunk 1-NFE CARD streaming (seed 11), mirroring the KV-cache
    loop of `infer_ref.py::inference` (per-chunk clean-mel cache, cache
    trimmed to the last 100 positions once offset > 40)."""
    torch.manual_seed(11)
    bn = torch.randn(1, 8 * CHUNK, 256)
    prompts = torch.rand(1, 120, 80) * 2 - 1
    spks = torch.randn(1, 256)
    out = {"bn": bn, "prompts": prompts, "spks": spks}

    cache, kv_cache, offset = None, None, 0
    for q in range(8):
        noise = torch.randn(1, CHUNK, 80)
        u, kv_cache = model(
            noise, torch.ones(1), torch.zeros(1), cache=cache,
            cond=bn[:, q * CHUNK:(q + 1) * CHUNK], spks=spks,
            prompts=prompts, offset=offset, is_inference=True,
            kv_cache=kv_cache,
        )
        out[f"n{q}"] = noise
        out[f"u{q}"] = u
        cache = noise - u  # x = x - (t - r) * u with (t, r) = (1, 0)
        offset += CHUNK
        if offset > 40 and kv_cache[0][0].shape[2] > KV_CACHE_MAX_LEN:
            kv_cache = [
                (k[:, :, -KV_CACHE_MAX_LEN:, :], v[:, :, -KV_CACHE_MAX_LEN:, :])
                for k, v in kv_cache
            ]
    save_file(out, str(ckpt / "dit_stream_fixture.safetensors"))
    print("dit_stream_fixture.safetensors: 8 chunks of", tuple(out["u0"].shape))


def official_mel(y: torch.Tensor) -> torch.Tensor:
    """`MelSpectrogramFeatures` from `infer_ref.py`: magnitude STFT →
    slaney filterbank → dB → [-1, 1]. `y` [1, n] → mel [1, 80, frames]."""
    basis = torch.from_numpy(
        librosa_mel_fn(sr=16_000, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
    ).float()
    spec = torch.stft(
        y, 1024, hop_length=160, win_length=640,
        window=torch.hann_window(640), center=True, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True,
    )
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
    spec = basis @ spec
    min_level = float(np.exp(-115 / 20 * np.log(10)))
    spec = 20 * torch.log10(torch.clamp(spec, min=min_level)) - 20
    return torch.clamp(2 * ((spec + 115) / 115) - 1, -1, 1)


@torch.no_grad()
def gen_copysyn_fixture(ckpt: Path):
    """Official mel of test.wav + Vocos (TorchScript) copy-synthesis."""
    y = load_test_wav(ckpt).unsqueeze(0)
    mel_raw = official_mel(y).squeeze(0).T.contiguous()  # [frames, 80]
    mel01 = (mel_raw + 1) / 2  # vocoder input domain (infer_ref.py)
    vocos = torch.jit.load(ckpt / "vocos.pt", map_location="cpu")
    wav_ref = vocos.decode(mel01.T.unsqueeze(0)).squeeze(0).contiguous()
    save_file(
        {"mel_raw": mel_raw, "mel01": mel01, "wav_ref": wav_ref},
        str(ckpt / "copysyn_fixture.safetensors"),
    )
    print(
        "copysyn_fixture.safetensors: mel", tuple(mel_raw.shape),
        "wav", tuple(wav_ref.shape),
    )


def official_fbank(wav: torch.Tensor) -> torch.Tensor:
    """`extract_fbanks(..., frame_shift=10)` from `infer_ref.py`."""
    return kaldi.fbank(
        (wav * (1 << 15)).unsqueeze(0), frame_length=25, frame_shift=10,
        snip_edges=True, num_mel_bins=80, energy_floor=0.0, dither=0.0,
        sample_frequency=16_000,
    ).unsqueeze(0)


@torch.no_grad()
def gen_asr_fixtures(ckpt: Path):
    """Fast-U2++ references from the TorchScript checkpoint:

    * `asr_chunk0_fixture`: the first decoding window (23 raw fbank
      frames, no CMVN) through `encoder.embed` (subsampling ×4 + ×√d
      scale), the sinusoidal pos-emb, conformer layer 0 alone, and the
      full `forward_encoder_chunk` chunk-0 output;
    * `pipeline_ref`: the full-utterance kaldi fbank and the BNFs from
      the official chunked decode loop (`infer_ref.py`:
      decoding_chunk_size 5, 2 left chunks, ×4 subsampling, context 7),
      upsampled ×4 with align_corners linear interpolation.
    """
    asr = torch.jit.load(ckpt / "fastu2pp.pt", map_location="cpu")
    fbanks = official_fbank(load_test_wav(ckpt))

    # --- chunk-0 stage-by-stage references -------------------------------
    dcs, nlc, sub, ctx = 5, 2, 4, 7  # decode loop constants (infer_ref.py)
    decoding_window = (dcs - 1) * sub + ctx  # 23
    fb23 = fbanks[:, :decoding_window].contiguous()
    masks = torch.ones(1, 1, decoding_window, dtype=torch.bool)
    # NOTE: fed raw (pre-CMVN) on purpose — mirrors `FastU2pp::subsample`.
    x_embed, pos, _ = asr.encoder.embed(fb23, masks, 0)
    att_mask = torch.ones(1, 1, x_embed.shape[1], dtype=torch.bool)
    zero_cache = torch.zeros(0, 0, 0, 0)
    layer0, *_ = getattr(asr.encoder.encoders, "0")(
        x_embed, att_mask, pos, att_mask, zero_cache, zero_cache
    )
    bn0, _, _ = asr.forward_encoder_chunk(
        fb23, 0, dcs * nlc, zero_cache, zero_cache
    )
    save_file(
        {
            "fb23": fb23, "x_embed": x_embed, "embed_ref": x_embed.clone(),
            "pos": pos, "layer0_ref": layer0, "bn0_ref": bn0,
        },
        str(ckpt / "asr_chunk0_fixture.safetensors"),
    )
    print("asr_chunk0_fixture.safetensors: bn0", tuple(bn0.shape))

    # --- full-utterance chunked decode loop ------------------------------
    stride = sub * dcs
    required_cache_size = dcs * nlc
    att_cache = cnn_cache = zero_cache
    offset, chunks = 0, []
    for i in range(0, fbanks.shape[1], stride):
        chunk = fbanks[:, i:i + decoding_window, :]
        if chunk.shape[1] < required_cache_size:
            pad = required_cache_size - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad))
        out, att_cache, cnn_cache = asr.forward_encoder_chunk(
            chunk, offset, required_cache_size, att_cache, cnn_cache
        )
        offset += out.size(1)
        chunks.append(out)
    bn = torch.cat(chunks, dim=1).transpose(1, 2)
    bn = torch.nn.functional.interpolate(
        bn, size=bn.shape[2] * sub, mode="linear", align_corners=True
    ).transpose(1, 2).contiguous()
    save_file(
        {"fbank_ref": fbanks.contiguous(), "bn_ref": bn},
        str(ckpt / "pipeline_ref.safetensors"),
    )
    print(
        "pipeline_ref.safetensors: fbank", tuple(fbanks.shape),
        "bn", tuple(bn.shape),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt"))
    parser.add_argument(
        "--meanvc-repo", type=Path, default=None,
        help="existing ASLP-lab/MeanVC clone (default: shallow-clone to a tempdir)",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    gen_copysyn_fixture(args.ckpt)
    gen_asr_fixtures(args.ckpt)
    DiT = import_official_dit(args.meanvc_repo)
    model = load_official_dit(DiT, args.ckpt)
    gen_dit_fixture(model, args.ckpt)
    gen_dit_stream_fixture(model, args.ckpt)
    print("done — run: cargo test --release --test golden")


if __name__ == "__main__":
    main()
