"""
Dataset + augmentation for ESC-50 log-mel spectrograms.

Augmentation policy (train only), all in the spectrogram domain so it runs on
the cached features with no audio re-decoding:
  * random time roll        (circular shift along time  -> ~time-shift invariance)
  * SpecAugment             (time & frequency masking)
  * mixup / Between-Class   (applied at batch level in the training loop)

Normalization is global (mean/std) computed on the TRAINING folds only and
passed in, so there is no train/val leakage.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class ESC50Spec(Dataset):
    def __init__(self, X, y, mean, std, train=True,
                 time_roll=True, spec_aug=True,
                 freq_mask=24, time_mask=48, n_freq_masks=2, n_time_masks=2,
                 seed=0):
        """
        X    : float32 [N, M, T]  log-mel spectrograms
        y    : int64   [N]        labels
        mean, std : scalars from training folds (global standardization)
        """
        self.X = X
        self.y = y.astype(np.int64)
        self.mean = float(mean)
        self.std = float(std) if std > 1e-6 else 1.0
        self.train = train
        self.time_roll = time_roll
        self.spec_aug = spec_aug
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def _augment(self, spec):
        M, T = spec.shape
        # random circular time shift
        if self.time_roll:
            shift = int(self.rng.integers(0, T))
            spec = np.roll(spec, shift, axis=1)
        # SpecAugment: frequency masks
        if self.spec_aug:
            fill = spec.min()
            for _ in range(self.n_freq_masks):
                f = int(self.rng.integers(0, self.freq_mask + 1))
                if f > 0:
                    f0 = int(self.rng.integers(0, max(1, M - f)))
                    spec[f0:f0 + f, :] = fill
            for _ in range(self.n_time_masks):
                t = int(self.rng.integers(0, self.time_mask + 1))
                if t > 0:
                    t0 = int(self.rng.integers(0, max(1, T - t)))
                    spec[:, t0:t0 + t] = fill
        return spec

    def __getitem__(self, idx):
        spec = self.X[idx].copy()
        if self.train:
            spec = self._augment(spec)
        spec = (spec - self.mean) / self.std
        # add channel dim -> [1, M, T]
        return torch.from_numpy(spec).unsqueeze(0), int(self.y[idx])


def mixup_batch(x, y, num_classes, alpha=0.3, generator=None):
    """
    Standard mixup applied on a batch. Returns mixed inputs and soft targets.
      x : [B, 1, M, T]     y : [B] long
    """
    if alpha <= 0:
        y1h = torch.zeros(x.size(0), num_classes, device=x.device)
        y1h.scatter_(1, y.view(-1, 1), 1.0)
        return x, y1h
    lam = float(np.random.default_rng().beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device, generator=generator)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    y1h = torch.zeros(x.size(0), num_classes, device=x.device)
    y1h.scatter_(1, y.view(-1, 1), 1.0)
    mixed_y = lam * y1h + (1.0 - lam) * y1h[perm]
    return mixed_x, mixed_y


def compute_norm_stats(X_train):
    """Global mean/std over the training split only (leakage-safe)."""
    return float(X_train.mean()), float(X_train.std())
