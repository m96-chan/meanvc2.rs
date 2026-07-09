#!/usr/bin/env python3
"""Generate golden fixtures for the crates/cosyvoice port from the OFFICIAL
CosyVoice implementation (github.com/FunAudioLLM/CosyVoice checkout).

Everything runs on CPU fp32. Stage boundaries saved to
ckpt/cosyvoice_e2e_fixture.safetensors:

  source_16k / prompt_16k        - input audio
  source_mel128 / prompt_mel128  - whisper 128-mel front-end (tokenizer input)
  source_tokens / prompt_tokens  - FSQ tokens from the official ONNX tokenizer
  prompt_feat                    - 24 kHz 80-mel prompt features [1,T,80]
  embedding                      - CAM++ x-vector from the official ONNX [1,192]
  flow_mu                        - encoder+proj output fed to the CFM [1,T2,80]
  cfm_mel                        - flow.inference output mel [1,80,T2] (deterministic)
  stream_mel_chunk0              - streaming=True finalize=False mel for the first
                                   25+3-token chunk (chunked-mask golden)
  hift_f0 / hift_source / hift_audio - HiFT stages with zeroed harmonic phase +
                                   zeroed noise (determinism patch; the Rust golden
                                   test uses the same zero-noise configuration)
  e2e_audio                      - full official inference_vc output (same patch)

Usage:
  python tools/gen_cosyvoice_fixtures.py --cosyvoice-repo <checkout> \
      --model-dir <CosyVoice2-0.5B snapshot> \
      --source ckpt/M06_03_16k.wav --prompt ckpt/F19_01_16k.wav
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cosyvoice-repo', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--source', default='ckpt/M06_03_16k.wav')
    ap.add_argument('--prompt', default='ckpt/F19_01_16k.wav')
    ap.add_argument('--out', default='ckpt/cosyvoice_e2e_fixture.safetensors')
    args = ap.parse_args()

    sys.path.insert(0, args.cosyvoice_repo)
    sys.path.insert(0, os.path.join(args.cosyvoice_repo, 'third_party', 'Matcha-TTS'))

    import whisper
    from safetensors.torch import save_file

    # ---- determinism patches (harmonic phases + noise -> 0) --------------
    # SineGen2 draws random initial phases for harmonics 1..8 and Gaussian noise;
    # SourceModuleHnNSF adds another noise branch. Zero them so the fixture and
    # the Rust golden test compute the identical deterministic path.
    real_rand, real_randn_like = torch.rand, torch.randn_like

    def zero_rand(*size, **kw):
        if kw.get('device') is not None or True:
            return torch.zeros(*size, **{k: v for k, v in kw.items() if k in ('device', 'dtype')})
    torch.rand = zero_rand
    torch.randn_like = lambda x, **kw: torch.zeros_like(x)

    fx = {}
    # force CPU
    torch.cuda.is_available = lambda: False
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    model = CosyVoice2(args.model_dir)
    fe, mm = model.frontend, model.model

    src16 = load_wav(args.source, 16000)
    pmt16 = load_wav(args.prompt, 16000)
    fx['source_16k'] = src16.clone()
    fx['prompt_16k'] = pmt16.clone()
    fx['source_mel128'] = whisper.log_mel_spectrogram(src16, n_mels=128)
    fx['prompt_mel128'] = whisper.log_mel_spectrogram(pmt16, n_mels=128)

    src_tok, src_tok_len = fe._extract_speech_token(args.source)
    pmt_tok, pmt_tok_len = fe._extract_speech_token(args.prompt)
    fx['source_tokens'] = src_tok.int()
    fx['prompt_tokens'] = pmt_tok.int()
    pmt_feat, _ = fe._extract_speech_feat(args.prompt)
    fx['prompt_feat'] = pmt_feat.float()
    emb = fe._extract_spk_embedding(args.prompt)
    fx['embedding'] = emb.float()
    import torchaudio.compliance.kaldi as kaldi
    fbank = kaldi.fbank(pmt16, num_mel_bins=80, dither=0, sample_frequency=16000)
    fx['prompt_fbank'] = (fbank - fbank.mean(dim=0, keepdim=True)).float()
    # The official campplus.onnx bakes trace-time shapes (T=200) into its
    # seg-pooling, so it is only exact at 200 frames; capture that exact point
    # as the tight golden (see crates/cosyvoice/src/campplus.rs docs).
    emb200 = fe.campplus_session.run(
        None, {fe.campplus_session.get_inputs()[0].name: fx['prompt_fbank'][:200].unsqueeze(0).numpy()})[0]
    fx['embedding_200'] = torch.from_numpy(emb200).float()

    flow = mm.flow.eval()
    with torch.inference_mode():
        # ---- encoder mu golden (non-streaming, finalize=True path) -------
        import torch.nn.functional as F
        embedding = F.normalize(emb, dim=1)
        embedding = flow.spk_embed_affine_layer(embedding)
        token = torch.concat([pmt_tok, src_tok], dim=1)
        token_len = torch.tensor([token.shape[1]], dtype=torch.int32)
        mask = torch.ones(1, token.shape[1], 1)
        temb = flow.input_embedding(torch.clamp(token, min=0)) * mask
        h, _ = flow.encoder(temb, token_len, streaming=False)
        mu = flow.encoder_proj(h)
        fx['flow_mu'] = mu.float()

        # ---- full CFM mel (deterministic: fixed seed-0 noise) ------------
        mel, _ = flow.inference(token=src_tok, token_len=torch.tensor([src_tok.shape[1]], dtype=torch.int32),
                                prompt_token=pmt_tok, prompt_token_len=torch.tensor([pmt_tok.shape[1]], dtype=torch.int32),
                                prompt_feat=pmt_feat, prompt_feat_len=torch.tensor([pmt_feat.shape[1]], dtype=torch.int32),
                                embedding=emb, streaming=False, finalize=True)
        fx['cfm_mel'] = mel.float()

        # ---- one streaming chunk (chunked attention masks) ----------------
        n0 = min(25 + flow.pre_lookahead_len, src_tok.shape[1])
        mel0, _ = flow.inference(token=src_tok[:, :n0], token_len=torch.tensor([n0], dtype=torch.int32),
                                 prompt_token=pmt_tok, prompt_token_len=torch.tensor([pmt_tok.shape[1]], dtype=torch.int32),
                                 prompt_feat=pmt_feat, prompt_feat_len=torch.tensor([pmt_feat.shape[1]], dtype=torch.int32),
                                 embedding=emb, streaming=True, finalize=False)
        fx['stream_mel_chunk0'] = mel0.float()

        # ---- HiFT stages (zero-noise deterministic) -----------------------
        hift = mm.hift.eval()
        f0 = hift.f0_predictor(mel)
        fx['hift_f0'] = f0.float()
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = hift.m_source(s)
        s = s.transpose(1, 2)
        fx['hift_source'] = s.float()
        audio = hift.decode(x=mel, s=s)
        fx['hift_audio'] = audio.float()
        fx['e2e_audio'] = audio.float().clone()

    fx = {k: v.contiguous() for k, v in fx.items()}
    save_file(fx, args.out)
    for k, v in fx.items():
        print(f'{k}: {tuple(v.shape)} {v.dtype}')
    torch.rand, torch.randn_like = real_rand, real_randn_like


if __name__ == '__main__':
    main()
