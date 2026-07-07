#!/usr/bin/env python3
"""Generate per-stage golden fixtures for the Seed-VC port (issue #50).

Drives the official implementation (whisper-small + wavenet + bigvgan
preset) on a short source/target pair and records every stage boundary,
with the CFM initial noise captured so the ODE trajectory is exactly
reproducible:

    seedvc_e2e_fixture.safetensors:
      source_22k, ref_22k                 inputs (22 050 Hz, float32)
      s_alt, s_ori                        whisper-small encoder features (50 Hz)
      mel, mel2                           80-bin mel @ 22 050 Hz (hop 256)
      feat2                               CAM++ kaldi fbank input (16 k, CMN)
      style2                              CAM++ embedding [1, 192]
      cond, prompt_condition              length-regulator outputs
      cfm_noise                           initial noise of cfm.inference
      vc_mel                              CFM output mel (10 steps, cfg 0.7)
      vc_wave                             BigVGAN output waveform

Whisper runs in fp32 here (the official inference uses fp16; the port
is fp32 by project policy, so the goldens are fp32-parity like the
X-VC fixtures).

    python tools/gen_seedvc_fixtures.py --seedvc-dir <clone> \
        --source ckpt/ref_trimmed.wav --target ckpt/ref_stage1.wav --out ckpt
"""
import argparse
import os
import sys
import types

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seedvc-dir", required=True)
    ap.add_argument("--source", default="ckpt/ref_trimmed.wav")
    ap.add_argument("--target", default="ckpt/ref_stage1.wav")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default="ckpt")
    args = ap.parse_args()

    os.chdir(args.seedvc_dir)
    sys.path.insert(0, args.seedvc_dir)

    # fp32 whisper for fp32-parity goldens.
    import transformers

    orig_from_pretrained = transformers.WhisperModel.from_pretrained.__func__

    def fp32_from_pretrained(cls, *a, **kw):
        kw["torch_dtype"] = torch.float32
        return orig_from_pretrained(cls, *a, **kw)

    transformers.WhisperModel.from_pretrained = classmethod(fp32_from_pretrained)

    import inference as inf

    ns = types.SimpleNamespace(
        fp16=False, checkpoint=None, config=None, f0_condition=False
    )
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = (
        inf.load_models(ns)
    )
    device = inf.device
    sr = mel_fn_args["sampling_rate"]
    assert sr == 22050

    import librosa
    import torchaudio

    torch.set_grad_enabled(False)

    n = int(args.seconds * sr)
    source_audio = torch.tensor(librosa.load(args.source, sr=sr)[0][:n]).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(librosa.load(args.target, sr=sr)[0][:n]).unsqueeze(0).float().to(device)

    converted_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    ori_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    s_alt = semantic_fn(converted_16k)
    s_ori = semantic_fn(ori_16k)

    mel = mel_fn(source_audio)
    mel2 = mel_fn(ref_audio)
    target_lengths = torch.LongTensor([mel.size(2)]).to(device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(device)

    import torchaudio.compliance.kaldi as kaldi

    feat2 = kaldi.fbank(ori_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    cond, *_ = model.length_regulator(s_alt, ylens=target_lengths, n_quantizers=3, f0=None)
    prompt_condition, *_ = model.length_regulator(
        s_ori, ylens=target2_lengths, n_quantizers=3, f0=None
    )
    cat_condition = torch.cat([prompt_condition, cond], dim=1)

    # Capture the CFM initial noise deterministically.
    captured = {}
    real_randn = torch.randn

    def capturing_randn(*a, **kw):
        t = real_randn(*a, **kw)
        if "noise" not in captured:
            captured["noise"] = t.detach().clone()
        return t

    torch.manual_seed(42)
    torch.randn = capturing_randn
    try:
        vc_mel = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(device),
            mel2,
            style2,
            None,
            10,
            inference_cfg_rate=0.7,
        )
    finally:
        torch.randn = real_randn
    vc_mel = vc_mel[:, :, mel2.size(2):]
    vc_wave = vocoder_fn(vc_mel).squeeze(1)

    from safetensors.torch import save_file

    out = {
        "source_22k": source_audio.cpu(),
        "ref_22k": ref_audio.cpu(),
        "s_alt": s_alt.cpu(),
        "s_ori": s_ori.cpu(),
        "mel": mel.cpu(),
        "mel2": mel2.cpu(),
        "feat2": feat2.cpu(),
        "style2": style2.cpu(),
        "cond": cond.cpu(),
        "prompt_condition": prompt_condition.cpu(),
        "cfm_noise": captured["noise"].cpu(),
        "vc_mel": vc_mel.cpu(),
        "vc_wave": vc_wave.cpu(),
    }
    out = {k: v.to(torch.float32).contiguous() for k, v in out.items()}
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", args.out,
        "seedvc_e2e_fixture.safetensors",
    ) if not os.path.isabs(args.out) else os.path.join(args.out, "seedvc_e2e_fixture.safetensors")
    save_file(out, os.path.abspath(path))
    for k, v in out.items():
        print(k, tuple(v.shape))
    print("wrote", os.path.abspath(path))


if __name__ == "__main__":
    main()
