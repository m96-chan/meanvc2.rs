#!/usr/bin/env python3
"""Exports the official MeanVC v1 voice-print model (WavLM-Large +
ECAPA_TDNN_SMALL speaker verification, `wavlm_large_finetune.pth`) to ONNX
for the Rust `backends::WavLmSv` (feature `wavlm`) — issue #15.

The s3prl wrapper is not ONNX-traceable (list handling / pad_sequence /
dynamic shapes), so this script (1) monkeypatches the s3prl expert forward
with a batch-1 equivalent (verified: cosine 1.0000 vs the original) and
(2) exports with a FIXED 5 s input (80 000 samples); the Rust backend
tiles/crops reference audio to that length.

Prereqs: the official repo cloned (for src.runtime.speaker_verification),
`wavlm_large_finetune.pth` (Google Drive, see the official README), and
pip: torch, s3prl (via torch.hub on first run), soundfile, onnx,
onnxruntime, safetensors.

Usage:
    python3 tools/export_wavlm_onnx.py <official_repo_dir> <wavlm_ckpt.pth> \
        [out.onnx=ckpt/wavlm_sv.onnx] [golden_wav=ckpt/test.wav]

Also regenerates ckpt/wavlm_golden.safetensors (tiled input + reference
embedding) used by the Rust golden test.
"""

import sys
import types
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

FIX = 80_000  # 5 s at 16 kHz

official = sys.argv[1]
ckpt = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "ckpt/wavlm_sv.onnx"
golden_wav = sys.argv[4] if len(sys.argv) > 4 else "ckpt/test.wav"

# s3prl (torch.hub) is incompatible with modern torchaudio; shim the
# removed APIs its import graph touches.
import torchaudio  # noqa: E402

torchaudio.set_audio_backend = lambda *a, **k: None
sox = types.ModuleType("torchaudio.sox_effects")
sox.apply_effects_tensor = None
sox.apply_effects_file = None
sys.modules["torchaudio.sox_effects"] = sox
torchaudio.sox_effects = sox

sys.path.insert(0, official)
from src.runtime.speaker_verification.verification import init_model  # noqa: E402

model = init_model("wavlm_large", ckpt)
model.eval()

wav = torch.randn(1, FIX)
with torch.no_grad():
    ref = model(wav)

# Batch-1, fixed-length equivalent of the s3prl expert forward (avoids
# pad_sequence and shape-dependent list ops). Hooks that assemble
# "hidden_states" still fire, so the feature path is unchanged.
Expert = type(model.feature_extract)


def patched_forward(self, wavs):
    w = F.layer_norm(wavs[0], wavs[0].shape)
    features, _ = self.model.extract_features(
        w.unsqueeze(0), padding_mask=None, mask=False
    )
    return {"default": features}


Expert.forward = patched_forward

with torch.no_grad():
    patched = model(wav)
cos = F.cosine_similarity(ref, patched).item()
print(f"patched vs original cosine: {cos:.6f}")
assert cos > 0.9999, "patched forward diverges from the original"

torch.onnx.export(
    model,
    (wav,),
    out,
    input_names=["wav"],
    output_names=["embedding"],
    opset_version=17,
    dynamo=False,
)
print(f"exported {out}")

# Validate with onnxruntime and dump the Rust golden fixture.
import onnxruntime as ort  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
check = sess.run(None, {"wav": wav.numpy()})[0]
cos2 = F.cosine_similarity(ref, torch.from_numpy(check)).item()
print(f"onnxruntime vs eager cosine: {cos2:.6f}")
assert cos2 > 0.9999

audio, sr = sf.read(golden_wav)
audio = audio.astype(np.float32)
tiled = np.tile(audio, FIX // len(audio) + 1)[:FIX]
emb = sess.run(None, {"wav": tiled[None, :]})[0][0]
save_file(
    {"wav_tiled": torch.from_numpy(tiled), "embedding_ref": torch.from_numpy(emb)},
    "ckpt/wavlm_golden.safetensors",
)
print("wrote ckpt/wavlm_golden.safetensors")
