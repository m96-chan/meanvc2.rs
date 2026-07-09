#!/usr/bin/env python3
"""Generate per-stage golden fixtures for the Vevo-Timbre port (issue #74).

Drives the official Amphion implementation (models/vc/vevo,
Vevo-Timbre / `inference_fm` path) on a short source/reference pair and
records every stage boundary, with the CFM initial noise captured so the
ODE trajectory is exactly reproducible — same technique as
gen_seedvc_fixtures.py.

    vevo_e2e_fixture.safetensors:
      src_24k, src_16k, ref_24k, ref_16k    inputs (float32)
      hubert_src_raw, hubert_ref_raw        HuBERT-large layer-18 features (50 Hz, pre-norm)
      hubert_src_norm, hubert_ref_norm      same, z-normalized (hubert_large_l18_mean_std.npz)
      hubert_norm_mean, hubert_norm_std     the norm stats themselves (1024-d)
      repcodec_src_enc, repcodec_ref_enc    RepCodec VocosBackbone encoder output, pre-VQ
      repcodec_src_codes, repcodec_ref_codes  int64 codebook indices [T]
      ref_mel                               extract_mel_feature(ref_24k), normalized (the CFM `prompt`)
      cond_full                             fmt_model.cond_emb(cat([ref_codes, src_codes]))
      cfm_noise                             initial randn of reverse_diffusion
      diff_estimator_cond_{x,t,cond,mask,out}    one isolated DiffLlama forward (the with-prompt, cond branch, first ODE step)
      diff_estimator_uncond_{x,t,cond,mask,out}  the matching CFG uncond branch (zeroed cond, no prompt)
      fm_mel                                reverse_diffusion output (32 steps, cfg=1.0, rescale_cfg=0.75)
      wave_out                              Vocos(fm_mel) waveform

    python tools/gen_vevo_fixtures.py --amphion-dir <clone> \
        --source ckpt/ref_trimmed.wav --target ckpt/ref_stage1_48k.wav --out ckpt
"""
import argparse
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amphion-dir", required=True)
    ap.add_argument("--source", default="ckpt/ref_trimmed.wav")
    ap.add_argument("--target", default="ckpt/ref_stage1_48k.wav")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", default="ckpt")
    args = ap.parse_args()

    source = os.path.abspath(args.source)
    target = os.path.abspath(args.target)
    out_dir = os.path.abspath(args.out)

    os.chdir(args.amphion_dir)
    sys.path.insert(0, args.amphion_dir)

    import numpy as np
    import librosa
    import torchaudio
    from safetensors.torch import save_file

    from models.vc.vevo.vevo_utils import VevoInferencePipeline, load_wav

    torch.set_grad_enabled(False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    root = "./ckpts/Vevo/models--amphion--Vevo/snapshots"
    snap = os.path.join(root, os.listdir(root)[0])
    pipe = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=os.path.join(snap, "tokenizer/vq8192"),
        fmt_cfg_path="./models/vc/vevo/config/Vq8192ToMels.json",
        fmt_ckpt_path=os.path.join(snap, "acoustic_modeling/Vq8192ToMels"),
        vocoder_cfg_path="./models/vc/vevo/config/Vocoder.json",
        vocoder_ckpt_path=os.path.join(snap, "acoustic_modeling/Vocoder"),
        device=device,
    )

    n = int(args.seconds * 24000)
    src_24k = torch.tensor(librosa.load(source, sr=24000)[0][:n]).unsqueeze(0).float().to(device)
    ref_24k = torch.tensor(librosa.load(target, sr=24000)[0][:n]).unsqueeze(0).float().to(device)
    src_16k = torchaudio.functional.resample(src_24k, 24000, 16000)
    ref_16k = torchaudio.functional.resample(ref_24k, 24000, 16000)

    fixtures = {
        "src_24k": src_24k.cpu(),
        "ref_24k": ref_24k.cpu(),
        "src_16k": src_16k.cpu(),
        "ref_16k": ref_16k.cpu(),
        "hubert_norm_mean": pipe.hubert_feat_norm_mean.cpu().float(),
        "hubert_norm_std": pipe.hubert_feat_norm_std.cpu().float(),
    }

    # ---- HuBERT-large layer 18 ----
    hubert_src_raw, _ = pipe.extract_hubert_feature(src_16k)
    hubert_ref_raw, _ = pipe.extract_hubert_feature(ref_16k)
    fixtures["hubert_src_raw"] = hubert_src_raw.cpu()
    fixtures["hubert_ref_raw"] = hubert_ref_raw.cpu()

    mean = pipe.hubert_feat_norm_mean.to(hubert_src_raw)
    std = pipe.hubert_feat_norm_std.to(hubert_src_raw)
    hubert_src_norm = (hubert_src_raw - mean) / std
    hubert_ref_norm = (hubert_ref_raw - mean) / std
    fixtures["hubert_src_norm"] = hubert_src_norm.cpu()
    fixtures["hubert_ref_norm"] = hubert_ref_norm.cpu()

    # ---- RepCodec (content-style tokenizer, fvq8192) ----
    tok = pipe.content_style_tokenizer
    enc_src = tok.encoder(hubert_src_norm.transpose(1, 2)).transpose(1, 2)
    enc_ref = tok.encoder(hubert_ref_norm.transpose(1, 2)).transpose(1, 2)
    fixtures["repcodec_src_enc"] = enc_src.cpu()
    fixtures["repcodec_ref_enc"] = enc_ref.cpu()

    src_codes, _ = pipe.extract_hubert_codec(tok, src_16k, duration_reduction=False)
    ref_codes, _ = pipe.extract_hubert_codec(tok, ref_16k, duration_reduction=False)
    fixtures["repcodec_src_codes"] = src_codes.cpu().long()
    fixtures["repcodec_ref_codes"] = ref_codes.cpu().long()

    # ---- mel of the reference (CFM prompt) ----
    ref_mel = pipe.extract_mel_feature(ref_24k)
    fixtures["ref_mel"] = ref_mel.cpu()

    # ---- cond embedding ----
    diffusion_input_codecs = torch.cat([ref_codes, src_codes], dim=1)
    fmt = pipe.fmt_model
    cond_full = fmt.cond_emb(diffusion_input_codecs)
    fixtures["cond_full"] = cond_full.cpu()

    # ---- capture one isolated DiffLlama forward per branch (first ODE step) ----
    captured = {}
    orig_forward = fmt.diff_estimator.forward

    def capturing_forward(x, diffusion_step, cond, x_mask, **kw):
        out = orig_forward(x, diffusion_step, cond, x_mask, **kw)
        if "cond" not in captured:
            captured["cond"] = (x.detach().clone(), diffusion_step.detach().clone(),
                                 cond.detach().clone(), x_mask.detach().clone(), out.detach().clone())
        elif "uncond" not in captured:
            captured["uncond"] = (x.detach().clone(), diffusion_step.detach().clone(),
                                   cond.detach().clone(), x_mask.detach().clone(), out.detach().clone())
        return out

    fmt.diff_estimator.forward = capturing_forward

    # ---- capture the CFM initial noise ----
    noise_box = {}
    real_randn = torch.randn

    def capturing_randn(*a, **kw):
        t = real_randn(*a, **kw)
        if "z" not in noise_box:
            noise_box["z"] = t.detach().clone()
        return t

    torch.randn = capturing_randn
    try:
        fm_mel = fmt.reverse_diffusion(
            cond=cond_full,
            prompt=ref_mel,
            n_timesteps=args.steps,
        )
    finally:
        torch.randn = real_randn
        fmt.diff_estimator.forward = orig_forward

    fixtures["cfm_noise"] = noise_box["z"].cpu()
    fixtures["fm_mel"] = fm_mel.cpu()

    for branch in ("cond", "uncond"):
        x, t, cond, mask, out = captured[branch]
        fixtures[f"diff_estimator_{branch}_x"] = x.cpu()
        fixtures[f"diff_estimator_{branch}_t"] = t.cpu()
        fixtures[f"diff_estimator_{branch}_cond"] = cond.cpu()
        fixtures[f"diff_estimator_{branch}_mask"] = mask.cpu()
        fixtures[f"diff_estimator_{branch}_out"] = out.cpu()

    # ---- vocoder ----
    wave_out = pipe.vocoder_model(fm_mel.transpose(1, 2)).cpu()
    fixtures["wave_out"] = wave_out

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "vevo_e2e_fixture.safetensors")
    save_file({k: v.contiguous() for k, v in fixtures.items()}, out_path)
    print(f"wrote {out_path} ({len(fixtures)} tensors)")
    for k, v in fixtures.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
