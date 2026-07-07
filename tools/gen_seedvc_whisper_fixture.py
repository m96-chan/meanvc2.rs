#!/usr/bin/env python3
"""Whisper-stage fixture for the Seed-VC port (#50): 16 k wave +
input_features + encoder output, CPU fp32. Usage:
    svenv/bin/python tools/gen_seedvc_whisper_fixture.py <seed-vc clone>
"""
import os, sys, torch
os.environ['HF_HUB_CACHE'] = os.path.join(sys.argv[1], 'checkpoints/hf_cache')
torch.set_grad_enabled(False)
from transformers import AutoFeatureExtractor, WhisperModel
from safetensors.torch import load_file, save_file
R='/home/m96-chan/project/m96-chan/babiniku.rs/'
fx = load_file(R+'ckpt/seedvc_e2e_fixture.safetensors')
import torchaudio
w16 = torchaudio.functional.resample(fx['source_22k'], 22050, 16000)
m = WhisperModel.from_pretrained('openai/whisper-small', torch_dtype=torch.float32)
del m.decoder
fe = AutoFeatureExtractor.from_pretrained('openai/whisper-small')
inp = fe([w16.squeeze(0).numpy()], return_tensors='pt', return_attention_mask=True, sampling_rate=16000)
feats = m._mask_input_features(inp.input_features, attention_mask=inp.attention_mask)
out = m.encoder(feats, return_dict=True).last_hidden_state
s = out[:, :w16.size(-1)//320+1]
save_file({'wave16k': w16.contiguous(), 'input_features': feats.contiguous().float(),
           's_alt': s.contiguous().float()}, R+'ckpt/seedvc_whisper_fixture.safetensors')
print('wave16k', tuple(w16.shape), 'features', tuple(feats.shape), 's', tuple(s.shape))
# e2eフィクスチャのs_altと一致するか(リサンプル・fp32経路の確認)
d=(s - fx['s_alt']).abs().max().item()
print('vs e2e fixture s_alt:', d)
