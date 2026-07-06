#!/usr/bin/env python3
"""Regenerates every X-VC parity fixture for the Phase-1 Rust port
(issue #30) deterministically into `ckpt/`.

Fixtures written (all consumed by the future `xvc_*` golden tests;
shapes below are for the default inputs — 4.01 s `out.wav` source,
3.84 s `ckpt/test.wav` reference, 640 ms mid-utterance chunk):

| file                              | contents                                          |
|-----------------------------------|---------------------------------------------------|
| `xvc_preprocess_fixture.safetensors` | raw int16 wavs + official preprocessing (volume-norm, 40 Hz biquad HP, pad-to-1280) + Whisper 128-mel log spectrogram |
| `xvc_tokenizer_fixture.safetensors`  | GLM-4-Voice tokenizer: mel -> pre-VQ hidden, VQ ids, post-VQ hidden, 50 Hz hidden, `embed_ids`, semantic-adapter output (chunk + full utterance, the latter spans 2 causal blocks) |
| `xvc_speaker_fixture.safetensors`    | reference wav -> Kaldi fbank-80 (mean-norm) -> ERes2Net 192-d embedding |
| `xvc_codec_fixture.safetensors`      | SAC acoustic branch: wav -> z / z_e / zq / codes (encode) and converter-latent -> wav (decode) |
| `xvc_converter_fixture.safetensors`  | MMDiT `AcousticConverter` single step on seeded random inputs (seed 5) |
| `xvc_chain_fixture.safetensors`      | one chunk-level streaming step (640/240/100/20 window, no padding) with every stage intermediate + crossfade bookkeeping |
| `xvc_e2e_fixture.safetensors`        | out.wav -> test.wav: offline, official streaming preset (2400/120/100/20) and the CPU preset (640/240/100/20) |
| `xvc_inventory.json`                 | module path -> tensor shape for every checkpoint (xvc.pt generator, GLM-4-Voice tokenizer, ERes2Net) |

Prerequisites:

* a clone of https://github.com/Jerrister/X-VC with the released
  checkpoints in place (pass it with `--xvc-repo`; default
  `~/tmp/xvc-recon/X-VC`):
  - `ckpts/xvc.pt`                    (https://huggingface.co/chenxie95/X-VC)
  - `glm-4-voice-tokenizer/`          (https://huggingface.co/zai-org/glm-4-voice-tokenizer;
                                       falls back to the HF hub if absent)
  - `pretrained/speech_eres2net_sv_en_voxceleb_16k/`
                                      (https://modelscope.cn/models/iic/speech_eres2net_sv_en_voxceleb_16k)
* its Python deps (torch, transformers 4.44.x, torchaudio, soundfile,
  soxr, hydra-core, omegaconf, audiotools, einops) + safetensors;
* `out.wav` at the repo root and `ckpt/test.wav`.

Determinism notes (verified by running twice and hashing the outputs):

* everything runs on CPU (`CUDA_VISIBLE_DEVICES` is cleared before
  torch is imported) in fp32;
* the only randomness is the seeded `torch.randn` in the converter
  fixture; every other reference is a pure function of the audio;
* the official preprocessing operates in float64 (soundfile read +
  `highpass_biquad`) and casts to float32 at the end — mirror that on
  the Rust side or budget the tolerance for it;
* `mask_target_condition=False` throughout (the `infer_single.py`
  default, and what the Phase-0 listening outputs used).

Usage:

    python3 tools/gen_xvc_fixtures.py [--ckpt ckpt] [--xvc-repo PATH]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Determinism: force CPU before torch initializes CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical 640 ms chunk: source samples [3200, 13440) = [200 ms, 840 ms),
# i.e. window i=2 of the CPU streaming preset (640/240/100/20 -> history
# 280 ms), the first window that needs no zero padding. 10240 = 8 * 1280
# keeps the `latent_hop_length` alignment the official code asserts.
CHUNK_START, CHUNK_LEN = 3200, 10240
# CPU streaming preset from the Phase-0 recon (RTF 0.72 on 8 threads).
CPU_PRESET = dict(chunk_ms=640, current_ms=240, future_ms=100, smooth_ms=20)
# Official preset (scripts/batch_infer_seedtts_stream.sh).
OFFICIAL_PRESET = dict(chunk_ms=2400, current_ms=120, future_ms=100, smooth_ms=20)
SEED_CONVERTER = 5


def i64(v: int) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.int64)


def setup_xvc(repo: Path):
    """chdir into the official clone (its config paths are relative) and
    load the generator on CPU with weight_norm folded, exactly like
    `bins/infer_single.py`."""
    repo = repo.expanduser().resolve()
    if not (repo / "ckpts/xvc.pt").exists():
        sys.exit(f"missing {repo}/ckpts/xvc.pt — see the usage header")
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    local_cfg = repo / "configs/xvc_local.yaml"
    if not local_cfg.exists():
        # The shipped config resolves the tokenizer via the HF hub; point
        # it at a local snapshot when one is present (idempotent).
        text = (repo / "configs/xvc.yaml").read_text()
        if (repo / "glm-4-voice-tokenizer").exists():
            text = text.replace(
                "local_ckpt: null", 'local_ckpt: "glm-4-voice-tokenizer"'
            )
        local_cfg.write_text(text)

    from bins.infer_utils import load_xvc  # noqa: PLC0415

    cfg, model, device = load_xvc(str(local_cfg), "ckpts/xvc.pt", 0, False)
    assert device.type == "cpu", "fixtures must be generated on CPU"
    return cfg, model


def whisper_features(model, wav_bt: torch.Tensor):
    """The exact mel front-end of `WhisperVQEncoderWrapper.extract_and_encode`:
    WhisperFeatureExtractor (128 mel, n_fft 400, hop 160), padded to the
    tokenizer stride (2 * 4 * 160 = 1280 samples).
    `wav_bt` [B, n] -> (input_features [B, 128, frames], attention_mask [B, frames])."""
    se = model.semantic_encoder
    feats = se.feature_extractor(
        wav_bt.cpu().numpy(),
        sampling_rate=16000,
        return_attention_mask=True,
        return_tensors="pt",
        padding="longest",
        pad_to_multiple_of=se.stride,
    )
    return feats["input_features"].float(), feats["attention_mask"]


def load_pair(cfg, model):
    """Official preprocessing + precomputed conditions for
    out.wav (source) -> ckpt/test.wav (reference)."""
    from bins.infer_utils import load_pair_as_tensors, precompute_conditions  # noqa: PLC0415

    source_wav, target_wav, target_wav_cond = load_pair_as_tensors(
        source_wav_path=str(REPO_ROOT / "out.wav"),
        target_wav_path=str(REPO_ROOT / "ckpt/test.wav"),
        cfg=cfg,
        device=torch.device("cpu"),
        latent_hop_length=int(cfg["latent_hop_length"]),
        mask_target_condition=False,
    )
    speaker_condition, frame_condition = precompute_conditions(
        model, target_wav, target_wav_cond
    )
    return source_wav, target_wav, target_wav_cond, speaker_condition, frame_condition


@torch.inference_mode()
def gen_preprocess_fixture(model, pair, ckpt: Path):
    """Stage 1 — preprocessing: raw wav -> processed wav (float64
    volume-norm + 40 Hz `highpass_biquad` + zero-pad to a 1280-sample
    multiple, cast to f32) -> Whisper log-mel features."""
    source_wav, target_wav = pair[0], pair[1]
    raw_src, sr_s = sf.read(REPO_ROOT / "out.wav", dtype="int16")
    raw_tgt, sr_t = sf.read(REPO_ROOT / "ckpt/test.wav", dtype="int16")
    assert sr_s == sr_t == 16_000

    mel_src, mask_src = whisper_features(model, source_wav.squeeze(1))
    chunk = source_wav[:, :, CHUNK_START : CHUNK_START + CHUNK_LEN].clone()
    mel_chunk, mask_chunk = whisper_features(model, chunk.squeeze(1))

    save_file(
        {
            "raw_source_i16": torch.from_numpy(raw_src),
            "raw_target_i16": torch.from_numpy(raw_tgt),
            "processed_source": source_wav.contiguous(),
            "processed_target": target_wav.contiguous(),
            "mel_source": mel_src.contiguous(),
            "mel_source_mask": mask_src.contiguous(),
            "chunk_wav": chunk.contiguous(),
            "mel_chunk": mel_chunk.contiguous(),
            "mel_chunk_mask": mask_chunk.contiguous(),
            "chunk_start": i64(CHUNK_START),
            "sample_rate": i64(16_000),
        },
        str(ckpt / "xvc_preprocess_fixture.safetensors"),
    )
    print(
        "xvc_preprocess_fixture.safetensors: processed_source",
        tuple(source_wav.shape), "mel_source", tuple(mel_src.shape),
    )


@torch.inference_mode()
def gen_tokenizer_fixture(model, pair, ckpt: Path):
    """Stage 2 — GLM-4-Voice tokenizer: mel -> 16 truncated Whisper
    layers (causal convs + block-causal attention, block 200 frames) ->
    avg-pool k=4 -> pre-VQ hidden -> argmin VQ ids -> post-VQ hidden
    (+ learned pos-emb 2), plus `embed_ids` and the semantic adapter.

    The `full_*` tensors run the whole 4.08 s source (205 pre-pool
    frames -> 2 causal blocks) so the Rust mask is exercised across a
    block boundary; the `chunk` tensors stay within one block."""
    import models.codec.sac.modules.semantic_encoder as se_mod  # noqa: PLC0415

    se = model.semantic_encoder  # WhisperVQEncoderWrapper
    out = {}

    orig_vq = se_mod.vector_quantize
    captured = {}

    def capturing_vq(inputs, codebook):
        captured["prevq"] = inputs.detach().clone()
        return orig_vq(inputs, codebook)

    def run(prefix: str, wav_bt: torch.Tensor):
        mel, mask = whisper_features(model, wav_bt)
        se_mod.vector_quantize = capturing_vq
        try:
            enc = se.encoder(input_features=mel, attention_mask=mask)
        finally:
            se_mod.vector_quantize = orig_vq
        ids = enc.quantized_token_ids
        out[f"{prefix}mel"] = mel.contiguous()
        out[f"{prefix}mel_mask"] = mask.contiguous()
        out[f"{prefix}hidden_50hz"] = enc.whisper_hidden_states_50hz.contiguous()
        out[f"{prefix}hidden_prevq"] = captured["prevq"].contiguous()
        out[f"{prefix}token_ids"] = ids.contiguous()
        out[f"{prefix}hidden_postvq"] = enc.last_hidden_state.contiguous()
        return ids

    source_wav = pair[0]
    chunk = source_wav[:, :, CHUNK_START : CHUNK_START + CHUNK_LEN].clone()
    ids = run("", chunk.squeeze(1))
    run("full_", source_wav.squeeze(1))

    sem_emb = se.embed_ids(ids)  # [B, T12.5, 1280]
    adapter_out = model.semantic_adapter(sem_emb.transpose(1, 2))  # [B, 1024, T50]
    out["sem_emb"] = sem_emb.contiguous()
    out["adapter_out"] = adapter_out.contiguous()

    save_file(out, str(ckpt / "xvc_tokenizer_fixture.safetensors"))
    print(
        "xvc_tokenizer_fixture.safetensors: token_ids", tuple(ids.shape),
        "hidden_prevq", tuple(out["hidden_prevq"].shape),
        "full_token_ids", tuple(out["full_token_ids"].shape),
        "adapter_out", tuple(adapter_out.shape),
    )


@torch.inference_mode()
def gen_speaker_fixture(model, pair, ckpt: Path):
    """Stage 3 — ERes2Net speaker encoder: processed reference wav ->
    Kaldi fbank-80 (dither 0, utterance mean-norm) -> 192-d embedding."""
    target_wav = pair[1]  # [1, 1, n]
    spk = model.speaker_encoder
    fbank = torch.stack([spk.feat_extractor(w) for w in target_wav])  # [1, T, 80]
    emb, _latent = spk(target_wav)
    save_file(
        {
            "ref_wav": target_wav.contiguous(),
            "fbank": fbank.contiguous(),
            "embedding": emb.contiguous(),
        },
        str(ckpt / "xvc_speaker_fixture.safetensors"),
    )
    print(
        "xvc_speaker_fixture.safetensors: fbank", tuple(fbank.shape),
        "embedding", tuple(emb.shape),
    )


@torch.inference_mode()
def gen_codec_fixture(model, pair, dec_in: torch.Tensor, ckpt: Path):
    """Stage 4 — SAC acoustic codec, encode and decode separately:

    * encode: chunk wav [1, 1, 10240] -> DAC encoder `z` [1, 1024, 32]
      -> FactorizedVectorQuantize (`z_e` 8-d projected latents, `codes`
      argmin ids over the L2-normalized 16384 x 8 codebook, `zq` the
      out-projected quantized latents);
    * decode: the converter output latent for the same chunk (`dec_in`,
      in-distribution) -> DAC/HiFiGAN decoder waveform.
    """
    source_wav = pair[0]
    chunk = source_wav[:, :, CHUNK_START : CHUNK_START + CHUNK_LEN].clone()
    z = model.acoustic_encoder(chunk)
    zq, codes, *_ = model.acoustic_quantizer(z)
    z_e = model.acoustic_quantizer.in_project(z)
    dec_out = model.acoustic_decoder(dec_in)
    save_file(
        {
            "chunk_wav": chunk.contiguous(),
            "z": z.contiguous(),
            "z_e": z_e.contiguous(),
            "zq": zq.contiguous(),
            "codes": codes.contiguous(),
            "dec_in": dec_in.contiguous(),
            "dec_out": dec_out.contiguous(),
        },
        str(ckpt / "xvc_codec_fixture.safetensors"),
    )
    print(
        "xvc_codec_fixture.safetensors: z", tuple(z.shape),
        "codes", tuple(codes.shape), "dec_out", tuple(dec_out.shape),
    )


@torch.inference_mode()
def gen_converter_fixture(model, ckpt: Path):
    """Stage 5 — MMDiT `AcousticConverter`, one step on fixed random
    inputs (seed 5): joint attention over [x-seq (RoPE) || mel-cond seq
    (own RoPE)], AdaLN-Zero from the 192-d speaker condition."""
    torch.manual_seed(SEED_CONVERTER)
    x = torch.randn(1, 1024, 40)  # acoustic latent, 50 Hz [B, C, T]
    frame_cond = torch.randn(1, 128, 150)  # target mel cond [B, mel, T_cond]
    spk = torch.randn(1, 192)  # speaker condition [B, D]
    out = model.acoustic_converter(x, frame_cond, spk)
    save_file(
        {
            "x": x, "frame_cond": frame_cond, "spk": spk,
            "out": out.contiguous(), "seed": i64(SEED_CONVERTER),
        },
        str(ckpt / "xvc_converter_fixture.safetensors"),
    )
    print("xvc_converter_fixture.safetensors: out", tuple(out.shape))


@torch.inference_mode()
def gen_chain_fixture(model, pair, ckpt: Path) -> torch.Tensor:
    """Stage 5b/6a — one chunk-level streaming step, exactly
    `bins/infer_utils.py::run_stream_chunk_forward` on the canonical
    640 ms window with real conditions, every intermediate dumped.
    Includes the driver bookkeeping of the 640/240/100/20 preset:
    `wav_current` is the emitted slice, `wav_tail` the crossfade buffer.
    Returns the converter output latent (reused as the codec decode input)."""
    source_wav, _, _, spk_cond, frame_cond = pair
    chunk = source_wav[:, :, CHUNK_START : CHUNK_START + CHUNK_LEN].clone()

    se = model.semantic_encoder
    feat = se.extract_and_encode(chunk.squeeze(1))["speech_tokens"]
    sem_emb = se.embed_ids(feat)
    sem_up = model.semantic_adapter(sem_emb.transpose(1, 2)).transpose(1, 2)
    z = model.acoustic_encoder(chunk)
    zq = model.acoustic_quantizer(z)[0]
    combined = torch.cat([sem_up, zq.transpose(1, 2)], dim=2)  # [B, T50, 2048]
    prenet_out = model.prenet(combined.transpose(1, 2), spk_cond)  # [B, 1024, T50]
    conv_out = model.acoustic_converter(prenet_out, frame_cond, spk_cond)
    wav_out = model.acoustic_decoder(conv_out)

    # Driver slices (history 280 ms, current 240 ms, smooth 20 ms @ 16 kHz).
    hist = (CPU_PRESET["chunk_ms"] - CPU_PRESET["current_ms"]
            - CPU_PRESET["future_ms"] - CPU_PRESET["smooth_ms"]) * 16
    cur = CPU_PRESET["current_ms"] * 16
    smooth = CPU_PRESET["smooth_ms"] * 16
    save_file(
        {
            "chunk_wav": chunk.contiguous(),
            "chunk_start": i64(CHUNK_START),
            "speaker_condition": spk_cond.contiguous(),
            "frame_condition": frame_cond.contiguous(),
            "token_ids": feat.contiguous(),
            "sem_adapter_out": sem_up.contiguous(),
            "acoustic_zq": zq.contiguous(),
            "prenet_out": prenet_out.contiguous(),
            "converter_out": conv_out.contiguous(),
            "wav_out": wav_out.contiguous(),
            "wav_current": wav_out[:, :, hist : hist + cur].clone(),
            "wav_tail": wav_out[:, :, hist + cur : hist + cur + smooth].clone(),
            "history_ms": i64(hist // 16), "current_ms": i64(cur // 16),
            "future_ms": i64(CPU_PRESET["future_ms"]), "smooth_ms": i64(smooth // 16),
        },
        str(ckpt / "xvc_chain_fixture.safetensors"),
    )
    print(
        "xvc_chain_fixture.safetensors: prenet_out", tuple(prenet_out.shape),
        "converter_out", tuple(conv_out.shape), "wav_out", tuple(wav_out.shape),
    )
    return conv_out


@torch.inference_mode()
def gen_e2e_fixture(model, pair, ckpt: Path):
    """Stage 6 — end to end (out.wav -> test.wav): offline, the official
    streaming preset (2400/120/100/20) and the CPU-viable preset
    (640/240/100/20), incl. the precomputed conditions."""
    from bins.infer_utils import run_offline, run_streaming  # noqa: PLC0415

    source_wav, target_wav, target_wav_cond, spk_cond, frame_cond = pair
    offline = run_offline(model, source_wav, target_wav, target_wav_cond)
    out = {
        "source_wav": source_wav.contiguous(),
        "target_wav": target_wav.contiguous(),
        "speaker_condition": spk_cond.contiguous(),
        "frame_condition": frame_cond.contiguous(),
        "offline_out": offline.contiguous(),
    }
    for name, preset in (("official", OFFICIAL_PRESET), ("cpu", CPU_PRESET)):
        recon, _lat = run_streaming(
            model, source_wav, spk_cond, frame_cond, sample_rate=16_000, **preset
        )
        out[f"stream_{name}_out"] = recon.contiguous()
        for k, v in preset.items():
            out[f"stream_{name}_{k}"] = i64(v)
    save_file(out, str(ckpt / "xvc_e2e_fixture.safetensors"))
    print(
        "xvc_e2e_fixture.safetensors: offline_out", tuple(offline.shape),
        "stream_official_out", tuple(out["stream_official_out"].shape),
        "stream_cpu_out", tuple(out["stream_cpu_out"].shape),
    )


def _shape_map(sd: dict) -> dict:
    return {k: list(v.shape) for k, v in sd.items() if isinstance(v, torch.Tensor)}


def _summary(sd: dict) -> dict:
    counts = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            top = k.split(".", 1)[0]
            counts[top] = counts.get(top, 0) + v.numel()
    return counts


def gen_inventory(repo: Path, ckpt: Path):
    """Component inventory (module path -> shape) of the RAW checkpoints,
    i.e. before `remove_weight_norm` folding (`parametrizations.weight.
    original0/1` pairs still present) — what a converter script reads."""
    inv = {}

    sd = torch.load("ckpts/xvc.pt", map_location="cpu", weights_only=False)
    gen = sd["generator"]
    entry = {
        "top_level_keys": sorted(sd.keys()),
        "generator": _shape_map(gen),
        "generator_summary_params": _summary(gen),
    }
    for key in ("ema_generator", "discriminator"):
        if key in sd and isinstance(sd[key], dict):
            shapes = _shape_map(sd[key])
            entry[key] = {
                "num_tensors": len(shapes),
                "total_params": sum(int(np.prod(s)) for s in shapes.values()),
                "note": "training-only; tensor names mirror `generator` "
                        "(ema prefixed `ema_model.`)" if key == "ema_generator"
                        else "training-only (GAN discriminator)",
            }
    inv["ckpts/xvc.pt"] = entry
    del sd, gen

    tok = repo / "glm-4-voice-tokenizer/model.safetensors"
    if tok.exists():
        with safe_open(str(tok), framework="pt") as f:
            inv["glm-4-voice-tokenizer/model.safetensors"] = {
                k: list(f.get_slice(k).get_shape()) for k in f.keys()
            }

    eres = repo / "pretrained/speech_eres2net_sv_en_voxceleb_16k/pretrained_eres2net.ckpt"
    if eres.exists():
        inv[
            "pretrained/speech_eres2net_sv_en_voxceleb_16k/pretrained_eres2net.ckpt"
        ] = _shape_map(torch.load(str(eres), map_location="cpu", weights_only=True))

    path = ckpt / "xvc_inventory.json"
    path.write_text(json.dumps(inv, indent=1, sort_keys=True) + "\n")
    print(f"xvc_inventory.json: {len(inv)} checkpoints, "
          f"{sum(len(v.get('generator', v)) if isinstance(v, dict) else 0 for v in inv.values())} entries")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "ckpt")
    parser.add_argument(
        "--xvc-repo", type=Path, default=Path("~/tmp/xvc-recon/X-VC"),
        help="Jerrister/X-VC clone with checkpoints (see the usage header)",
    )
    args = parser.parse_args()
    ckpt = args.ckpt.resolve()

    torch.set_grad_enabled(False)
    cfg, model = setup_xvc(args.xvc_repo)
    pair = load_pair(cfg, model)

    gen_preprocess_fixture(model, pair, ckpt)
    gen_tokenizer_fixture(model, pair, ckpt)
    gen_speaker_fixture(model, pair, ckpt)
    gen_converter_fixture(model, ckpt)
    conv_out = gen_chain_fixture(model, pair, ckpt)
    gen_codec_fixture(model, pair, conv_out, ckpt)
    gen_e2e_fixture(model, pair, ckpt)
    gen_inventory(args.xvc_repo.expanduser().resolve(), ckpt)
    print("done — X-VC golden fixtures regenerated in", ckpt)


if __name__ == "__main__":
    main()
