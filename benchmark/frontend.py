"""
On-GPU audio front-end + waveform-domain augmentation for FRESCA-Net.

The log-mel transform runs on the GPU (torch.stft + a precomputed mel
filterbank matmul) so that energy-domain Between-Class mixing, random crops,
gain and noise can be applied to the raw waveform BEFORE the STFT -- which is
what makes BC mixing exact (log-mel mixing would break its energy-additive
semantics). The front-end has NO learnable parameters (mel bank / window /
coordinate map are buffers), so SWA only needs to average the backbone.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

SR = 44100


class LogMelFrontend(nn.Module):
    def __init__(self, sr=SR, n_fft=1024, hop=512, n_mels=128,
                 fmin=20, fmax=22050, coord=True):
        super().__init__()
        self.n_fft, self.hop, self.coord = n_fft, hop, coord
        mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                  fmin=fmin, fmax=fmax).astype(np.float32)
        self.register_buffer("mel_fb", torch.from_numpy(mel))          # [M, F]
        self.register_buffer("window", torch.hann_window(n_fft))
        cm = torch.linspace(-1, 1, n_mels).view(1, 1, n_mels, 1)       # freq coord
        self.register_buffer("coord_map", cm)

    def forward(self, wav):                        # wav: [B, L]
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.n_fft, window=self.window,
                          center=True, return_complex=True)            # [B, F, T]
        power = spec.real ** 2 + spec.imag ** 2
        mel = torch.matmul(self.mel_fb, power)                         # [B, M, T]
        logmel = torch.log(mel + 1e-6)
        # per-instance standardization (leakage-free, robust to BC shift)
        mean = logmel.mean(dim=(1, 2), keepdim=True)
        std = logmel.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        x = ((logmel - mean) / std).unsqueeze(1)                       # [B, 1, M, T]
        if self.coord:
            coord = self.coord_map.expand(x.size(0), 1, -1, x.size(3))
            x = torch.cat([x, coord], dim=1)                           # [B, 2, M, T]
        return x


# --------------------------- waveform augmentation ---------------------------
def random_crop(wav, crop_len, g):
    """Random crop each waveform to crop_len. wav: [B, L]."""
    B, L = wav.shape
    if L <= crop_len:
        return F.pad(wav, (0, crop_len - L))
    starts = torch.randint(0, L - crop_len + 1, (B,), device=wav.device, generator=g)
    idx = starts.view(B, 1) + torch.arange(crop_len, device=wav.device).view(1, -1)
    return torch.gather(wav, 1, idx)


def fixed_crops(wav, crop_len, n=3):
    """n evenly-spaced crops for test-time averaging. Returns list of [B, crop_len]."""
    B, L = wav.shape
    if L <= crop_len:
        w = F.pad(wav, (0, crop_len - L))
        return [w]
    starts = torch.linspace(0, L - crop_len, n).long()
    return [wav[:, s:s + crop_len] for s in starts]


def wave_shift(wav, max_frac, g):
    """Circular time-shift up to +/- max_frac of length, per sample."""
    B, L = wav.shape
    mx = int(max_frac * L)
    if mx == 0:
        return wav
    sh = torch.randint(-mx, mx + 1, (B,), device=wav.device, generator=g)
    idx = (torch.arange(L, device=wav.device).view(1, -1) - sh.view(B, 1)) % L
    return torch.gather(wav, 1, idx)


def wave_gain(wav, max_db, g):
    B = wav.size(0)
    db = (torch.rand(B, 1, device=wav.device, generator=g) * 2 - 1) * max_db
    return wav * (10.0 ** (db / 20.0))


def wave_noise(wav, snr_lo, snr_hi, p, g):
    B = wav.size(0)
    apply = torch.rand(B, 1, device=wav.device, generator=g) < p
    snr = torch.rand(B, 1, device=wav.device, generator=g) * (snr_hi - snr_lo) + snr_lo
    sig_p = wav.pow(2).mean(dim=1, keepdim=True).clamp(min=1e-10)
    noise_p = sig_p / (10.0 ** (snr / 10.0))
    noise = torch.randn(wav.shape, device=wav.device, generator=g) * noise_p.sqrt()
    return wav + noise * apply


def bc_mix(wav, y1h, g):
    """
    Energy-aware Between-Class mixing on the waveform (per-sample ratio).
      x = (p*x1 + (1-p)*x2) / sqrt(p^2 + (1-p)^2)   after RMS-normalizing each
    Returns mixed waveform and soft targets.
    """
    B = wav.size(0)
    rms = wav.pow(2).mean(dim=1, keepdim=True).clamp(min=1e-10).sqrt()
    wn = wav / rms
    perm = torch.randperm(B, device=wav.device, generator=g)
    p = torch.rand(B, 1, device=wav.device, generator=g)
    denom = (p * p + (1 - p) * (1 - p)).sqrt()
    mixed = (p * wn + (1 - p) * wn[perm]) / denom
    tgt = p * y1h + (1 - p) * y1h[perm]
    return mixed, tgt


def spec_augment(spec, g, n_fm=2, n_tm=2, fw=16, tw_frac=0.2, ramp=1.0):
    """
    SpecAugment on the (standardized) log-mel channel only. spec: [B, C, M, T].
    Masks are set to 0 (the per-instance mean after standardization). `ramp` in
    [0,1] scales mask widths for the warm-up schedule.
    """
    B, C, M, T = spec.shape
    x = spec.clone()
    fw = max(1, int(fw * ramp))
    tw = max(1, int(tw_frac * T * ramp))
    mrange = torch.arange(M, device=spec.device).view(1, M, 1)
    trange = torch.arange(T, device=spec.device).view(1, 1, T)
    ch0 = x[:, 0]                                     # only mask the log-mel channel
    for _ in range(n_fm):
        f = torch.randint(0, fw + 1, (B,), device=spec.device, generator=g)
        f0 = (torch.rand(B, device=spec.device, generator=g) * (M - f).clamp(min=1)).long()
        m = (mrange >= f0.view(B, 1, 1)) & (mrange < (f0 + f).view(B, 1, 1))
        ch0 = torch.where(m, torch.zeros_like(ch0), ch0)
    for _ in range(n_tm):
        t = torch.randint(0, tw + 1, (B,), device=spec.device, generator=g)
        t0 = (torch.rand(B, device=spec.device, generator=g) * (T - t).clamp(min=1)).long()
        m = (trange >= t0.view(B, 1, 1)) & (trange < (t0 + t).view(B, 1, 1))
        ch0 = torch.where(m, torch.zeros_like(ch0), ch0)
    x[:, 0] = ch0
    return x
