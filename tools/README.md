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
