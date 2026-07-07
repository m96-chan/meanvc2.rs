#!/usr/bin/env python3
"""Dump a single-estimator-forward fixture for the dit.rs port.

Runs one CFG-stacked (B=2) estimator forward at fixed t=0.5 using the
e2e fixture inputs, capturing internal activations for bisection.
"""
import os
import sys
import types

import torch

REPO = "/home/m96-chan/project/m96-chan/babiniku.rs"
SEEDVC = "/tmp/claude-1000/-home-m96-chan-project-m96-chan-meanvc2-rs/df9b8bd0-6579-4c56-8d40-c1c895ef728f/scratchpad/seed-vc"

os.chdir(SEEDVC)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, SEEDVC)

import transformers

orig_from_pretrained = transformers.WhisperModel.from_pretrained.__func__


def fp32_from_pretrained(cls, *a, **kw):
    kw["torch_dtype"] = torch.float32
    return orig_from_pretrained(cls, *a, **kw)


transformers.WhisperModel.from_pretrained = classmethod(fp32_from_pretrained)

import inference as inf

ns = types.SimpleNamespace(fp16=False, checkpoint=None, config=None, f0_condition=False)
model, *_ = inf.load_models(ns)
torch.set_grad_enabled(False)

from safetensors.torch import load_file, save_file

fx = load_file(os.path.join(REPO, "ckpt/seedvc_e2e_fixture.safetensors"))
cond = fx["cond"]
prompt_condition = fx["prompt_condition"]
mel2 = fx["mel2"]
style2 = fx["style2"]
noise = fx["cfm_noise"]

mu = torch.cat([prompt_condition, cond], dim=1)  # [1, 1032, 512]
T = mu.size(1)
prompt_len = mel2.size(2)

x = noise.clone()
x[..., :prompt_len] = 0
prompt_x = torch.zeros_like(x)
prompt_x[..., :prompt_len] = mel2

# CFG stack exactly like solve_euler
sx = torch.cat([x, x], dim=0)
sp = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
ss = torch.cat([style2, torch.zeros_like(style2)], dim=0)
sm = torch.cat([mu, torch.zeros_like(mu)], dim=0)
t = torch.tensor([0.5, 0.5])

dev = inf.device
sx, sp, ss, sm, t = (v.to(dev) for v in (sx, sp, ss, sm, t))

est = model.cfm.estimator.eval().float()

caps = {}


def hook(name):
    def f(_m, _inp, out):
        caps[name] = out.detach().float().clone()

    return f


hooks = [
    est.t_embedder.register_forward_hook(hook("t1")),
    est.t_embedder2.register_forward_hook(hook("t2")),
    est.cond_x_merge_linear.register_forward_hook(hook("merged")),
    est.transformer.register_forward_hook(hook("trans_out")),
    est.skip_linear.register_forward_hook(hook("skip_out")),
    est.conv1.register_forward_hook(hook("conv1_out")),
    est.wavenet.register_forward_hook(hook("wavenet_out")),
    est.final_layer.register_forward_hook(hook("final_out")),
    est.transformer.layers[0].register_forward_hook(hook("layer0_out")),
    est.transformer.layers[6].register_forward_hook(hook("layer6_out")),
]

out = est(sx, sp, torch.LongTensor([T]).to(dev), t, ss, sm)

for h in hooks:
    h.remove()

data = {
    "t": t,
    "out": out,
    **caps,
}
data = {k: v.to(torch.float32).cpu().contiguous() for k, v in data.items()}
save_file(data, os.path.join(REPO, "ckpt/seedvc_dit_stage_fixture.safetensors"))
for k, v in data.items():
    print(k, tuple(v.shape))
print("done")
