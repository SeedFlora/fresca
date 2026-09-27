"""
One-iteration end-to-end CUDA self-test (sub-second GPU touch, NOT training).
Validates that the whole FRESCA-Net path runs on the GPU before committing to a
long gated run: frontend STFT/mel, all waveform augmentations, SpecAugment,
AMP fwd/bwd, SWA parameter update, SWA BN recompute, and 3-crop TTA eval.
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import FRESCANet, count_params
from frontend import (LogMelFrontend, random_crop, fixed_crops,
                      wave_shift, wave_gain, wave_noise, bc_mix, spec_augment, SR)
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

dev = "cuda"
assert torch.cuda.is_available(), "no CUDA"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
W = np.load(os.path.join(CACHE, "waveforms.npy"), mmap_mode="r")
meta = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=True)
y = meta["y"]
g = torch.Generator(device=dev).manual_seed(0)

fe = LogMelFrontend(coord=True).to(dev).eval()
net = FRESCANet(in_ch=2).to(dev)
swa = AveragedModel(net)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")
crop_len = int(3.0 * SR)

t0 = time.time()
idx = np.arange(32)
wav = torch.from_numpy(np.ascontiguousarray(W[idx])).to(dev).float()
yb = torch.from_numpy(y[idx]).long().to(dev)
torch.cuda.synchronize()

net.train()
with torch.autocast("cuda", enabled=False):
    wav = random_crop(wav, crop_len, g)
    wav = wave_noise(wave_gain(wave_shift(wav, 0.2, g), 6.0, g), 20.0, 40.0, 0.3, g)
    y1h = torch.zeros(32, 50, device=dev); y1h.scatter_(1, yb.view(-1, 1), 1.0)
    wav, y1h = bc_mix(wav, y1h, g)
    spec = spec_augment(fe(wav), g, ramp=0.5)
with torch.autocast("cuda", enabled=True):
    out = net(spec)
    loss = -(y1h * F.log_softmax(out.float(), 1)).sum(1).mean()
scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
swa.update_parameters(net)

# SWA BN recompute (1 batch) + TTA eval
for m in swa.modules():
    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        m.reset_running_stats(); m.momentum = None
swa.train()
with torch.no_grad():
    swa(fe(random_crop(wav, crop_len, g)))
swa.module.eval()
with torch.no_grad():
    full = torch.from_numpy(np.ascontiguousarray(W[idx])).to(dev).float()
    probs = sum(torch.softmax(swa.module(fe(c)).float(), 1) for c in fixed_crops(full, crop_len, 3)) / 3
    pred = probs.argmax(1)
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"[selftest] OK  params={count_params(net)/1e6:.2f}M  loss={loss.item():.3f}  "
      f"spec={tuple(spec.shape)}  out={tuple(out.shape)}  pred[:5]={pred[:5].tolist()}")
print(f"[selftest] peak GPU mem = {peak:.2f} GB   wall = {time.time()-t0:.2f}s")
