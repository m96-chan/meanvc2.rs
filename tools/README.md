# tools — PyTorch parity fixtures & checkpoint conversion

Support scripts for the PyTorch parity campaign
([issue #8](https://github.com/m96-chan/meanvc2.rs/issues/8), v1 golden
suite in [issue #14](https://github.com/m96-chan/meanvc2.rs/issues/14)):
each generation script produces deterministic safetensors files containing
inputs and reference outputs computed by PyTorch; the Rust golden tests in
[`tests/golden.rs`](../tests/golden.rs) rebuild the same computation in
candle and compare.

Fixtures come in two tiers:

* **Committed** fixtures under `tools/fixtures/` — small and
  checkpoint-free, so `cargo test` exercises them without a Python
  environment. Regenerate after changing a script:

  ```sh
  pip install -r tools/requirements.txt
  python3 tools/gen_jvp_fixture.py
  cargo test --test golden
  ```

* **Checkpoint-dependent** fixtures under `ckpt/` (gitignored) — the
  MeanVC v1 parity suite against the official
  [ASLP-lab/MeanVC](https://github.com/ASLP-lab/MeanVC) release. The
  `v1_*` golden tests skip with a message when a file is missing, so the
  default suite stays green without checkpoints. To run them:

  ```sh
  # 1. download the official release into ckpt/:
  #    model_200ms.safetensors, config.json, vocos.pt, fastu2pp.pt, test.wav
  python3 tools/convert_official.py           # vocos/fastu2pp .pt -> .safetensors
  python3 tools/gen_v1_fixtures.py            # reference outputs (clones the
                                              # official repo; or --meanvc-repo PATH)
  cargo test --release --test golden
  ```

## Scripts

| Script | Fixture | Validates | Status |
|---|---|---|---|
| `gen_jvp_fixture.py` | `fixtures/jvp.safetensors` (committed) | `candle_core::forward_ad::jvp` vs `torch.func.jvp` on a mini DiT-like graph | ✅ |
| `convert_official.py` | `ckpt/{vocos,fastu2pp}.safetensors` | official TorchScript checkpoints → safetensors loadable by `backends::{Vocos, FastU2pp}` | ✅ |
| `gen_v1_fixtures.py` | `ckpt/dit_fixture.safetensors` | `v1::MeanVc1` single forward `u(x, r=0, t=1)` vs the official `dit_kvcache.py` DiT (< 1e-4) | ✅ |
| | `ckpt/dit_stream_fixture.safetensors` | 8-chunk CARD streaming with official KV-cache semantics (per-chunk < 1e-4) | ✅ |
| | `ckpt/copysyn_fixture.safetensors` | `v1::MelV1` vs the official mel chain (< 1e-4) and `backends::Vocos` copy-synthesis vs `vocos.pt` (wav < 5e-3) | ✅ |
| | `ckpt/asr_chunk0_fixture.safetensors` | Fast-U2++ chunk-0 stages: subsampling embed, sinusoidal pos-emb, conformer layer 0, full chunk output | ✅ |
| | `ckpt/pipeline_ref.safetensors` | `v1::KaldiFbank` vs `torchaudio.compliance.kaldi.fbank` (< 1e-3) and the full BNF path vs the official chunked decode loop (< 0.05) | ✅ |
| `gen_mel_fixture.py` | `fixtures/mel.safetensors` | `MelSpectrogram` vs torchaudio | planned (#8) |
| `convert_ecapa.py` | — | SpeechBrain ECAPA checkpoint → safetensors + golden output | planned (#8) |
| `gen_xvc_fixtures.py` | `ckpt/xvc_preprocess_fixture.safetensors` | X-VC preprocessing: volume-norm + 40 Hz highpass + pad-to-1280 (< 1e-5) and the Whisper 128-mel log spectrogram (< 1e-4) | ✅ |
| | `ckpt/xvc_tokenizer_fixture.safetensors` | GLM-4-Voice tokenizer: pre/post-VQ hidden states (< 1e-3), VQ token ids (exact), 50 Hz hidden, `embed_ids`, semantic adapter (< 1e-4) | ✅ |
| | `ckpt/xvc_speaker_fixture.safetensors` | ERes2Net: Kaldi fbank-80 mean-norm (< 1e-3) → 192-d embedding (cos > 0.9999) | ✅ |
| | `ckpt/xvc_codec_fixture.safetensors` | SAC acoustic codec: encode `z`/`z_e`/`zq` (< 1e-4) + codes (exact); decode latent → wav (< 5e-3) | ✅ |
| | `ckpt/xvc_converter_fixture.safetensors` | MMDiT `AcousticConverter` single step, seeded random inputs (< 1e-4) | ✅ |
| | `ckpt/xvc_chain_fixture.safetensors` | one 640 ms streaming chunk forward with every stage intermediate + crossfade slices | ✅ |
| | `ckpt/xvc_e2e_fixture.safetensors` | out.wav → test.wav end to end: offline + streaming (official 2400/120/100/20 and CPU 640/240/100/20 presets) | ✅ |
| | `ckpt/xvc_inventory.json` | module path → tensor shape for `xvc.pt` / GLM-4-Voice tokenizer / ERes2Net (porting reference) | ✅ |
| `convert_xvc_speaker.py` | `ckpt/xvc_speaker.safetensors` | ERes2Net speaker-encoder weights from `xvc.pt` (`speaker_encoder.model.*` — its BatchNorm running stats drifted from the ModelScope release during X-VC training) → `xvc::speaker::SpeakerEncoder`, verified by `crates/xvc/tests/golden_speaker.rs` | ✅ |

The X-VC fixtures ([issue #30](https://github.com/m96-chan/babiniku.rs/issues/30)
Phase 1) need the official [Jerrister/X-VC](https://github.com/Jerrister/X-VC)
clone with its checkpoints (`ckpts/xvc.pt`, `glm-4-voice-tokenizer/`,
`pretrained/speech_eres2net_sv_en_voxceleb_16k/`) — see the script header:

```sh
python3 tools/gen_xvc_fixtures.py --xvc-repo ~/tmp/xvc-recon/X-VC
```
