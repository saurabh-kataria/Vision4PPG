# lo/train_dinov3_bp_lastlayer.py
#!/usr/bin/env python3
import os, math, time, argparse, json
from typing import Optional, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel

# --- NEW: PEFT/LoRA ---
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def human_time(s):
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

class NpySignalDataset(Dataset):
    """
    PPG -> 2D image for DINOv3 (ViT) with robust filtering and multiple reps.

    reps:
      - "logpower":        3xF×T   (repeat of z(log-power))
      - "stft_complex3":   3xF×T   = [z(log|X|), z(cos φ), z(sin φ)]
      - "recurrence3":     3xN×N   Gaussian RPs of x, dx, d2x (downsample with rp_stride)

    Filtering (second implementation style):
      - Drops rows where X has NaN/Inf
      - Optional label bounds / custom label_valid_fn
      - Optional min_std to remove near-constant windows
      - Chunked scanning (mmap-safe)
      - Optional keep_idx cache

    Notes:
      • Per-rep z-norm applied inside builders; then ImageNet normalization.
      • For recurrence3, output is square NxN; keep window length fixed across dataset so N is constant.
    """
    def __init__(
        self,
        data_path: str,
        label_path: str,
        *,
        # Signal -> STFT params (for STFT reps)
        fs: int = 40,
        n_fft: int = 128,
        hop_length: int = 16,
        win_length: Optional[int] = None,
        clip_db: Optional[float] = 80.0,
        eps: float = 1e-6,

        # Training flags
        train: bool = False,
        spec_aug: bool = False,
        freq_mask_param: int = 4,
        time_mask_param: int = 4,

        # Representation selector
        rep: str = "recurrence3",  # "logpower" | "stft_complex3" | "recurrence3"

        # Recurrence-plot params
        rp_stride: int = 2,
        rp_sigma: float = 0.5,

        # Data hygiene
        label_min: Optional[float] = None,
        label_max: Optional[float] = None,
        label_valid_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        min_std: float = 0.0,
        check_chunk: int = 10000,
        keep_idx_cache: Optional[str] = None,
        verbose: bool = True,
    ):
        super().__init__()

        # --- load arrays ---
        self.X = np.load(data_path, mmap_mode="r")  # [N, L]
        y = np.load(label_path)
        if y.ndim > 1: y = y.squeeze()
        y = y.astype(np.float32, copy=False)
        if self.X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched N between X ({self.X.shape[0]}) and y ({y.shape[0]})")
        N = self.X.shape[0]

        # --- optional keep_idx cache ---
        keep_idx = None
        if keep_idx_cache and os.path.isfile(keep_idx_cache):
            try:
                cached = np.load(keep_idx_cache)
                if cached.ndim == 1 and cached.dtype.kind in "iu" and cached.max(initial=-1) < N:
                    keep_idx = cached.astype(np.int64, copy=False)
                    if verbose:
                        print(f"[Dataset] Loaded keep_idx cache: {keep_idx_cache} (kept {keep_idx.size}/{N})")
            except Exception:
                keep_idx = None

        # --- scan for valid rows if no cache ---
        if keep_idx is None:
            valid = np.ones(N, dtype=bool)

            # labels
            valid &= np.isfinite(y)
            if label_min is not None: valid &= (y >= label_min)
            if label_max is not None: valid &= (y <= label_max)
            if label_valid_fn is not None:
                mask = np.asarray(label_valid_fn(y), dtype=bool)
                if mask.shape != (N,):
                    raise ValueError("label_valid_fn must return boolean mask of shape [N]")
                valid &= mask

            # signals (NaN/Inf and optional min_std)
            for i in range(0, N, check_chunk):
                j = min(N, i + check_chunk)
                Xi = np.asarray(self.X[i:j])
                ok = np.isfinite(Xi).all(axis=1)
                if min_std > 0.0:
                    ok &= (Xi.std(axis=1) >= float(min_std))
                valid[i:j] &= ok

            keep_idx = np.nonzero(valid)[0].astype(np.int64)
            if keep_idx_cache:
                try:
                    os.makedirs(os.path.dirname(keep_idx_cache), exist_ok=True)
                except Exception:
                    pass
                try:
                    np.save(keep_idx_cache, keep_idx)
                    if verbose:
                        print(f"[Dataset] Saved keep_idx cache to {keep_idx_cache}")
                except Exception:
                    if verbose:
                        print(f"[Dataset] Warning: failed to save keep_idx cache")

        self.keep_idx = keep_idx
        self.n = int(self.keep_idx.size)
        self.y = y[self.keep_idx].astype(np.float32, copy=False)

        # --- core params ---
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.clip_db = clip_db
        self.eps = eps
        self._hann = torch.hann_window(self.win_length)

        self.train_flag = train
        self.spec_aug = spec_aug
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

        # reps
        self.rep = rep.lower()
        if self.rep not in {"logpower", "stft_complex3", "recurrence3"}:
            raise ValueError("rep must be one of {'logpower','stft_complex3','recurrence3'}")
        self.rp_stride = max(1, int(rp_stride))
        self.rp_sigma = float(rp_sigma)

        # DINOv3 (ImageNet) normalization
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.img_std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        if verbose:
            dropped = N - self.n
            pct = 0.0 if N == 0 else 100.0 * dropped / N
            print(f"[Dataset] kept {self.n}/{N} ({100-pct:.2f}%), dropped {dropped} ({pct:.2f}%)")

    def __len__(self):
        return self.n

    def _row(self, i: int) -> np.ndarray:
        return np.asarray(self.X[self.keep_idx[i]], dtype=np.float32)

    # ---------- helpers ----------
    @staticmethod
    def _z(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        m = img.mean(); s = img.std()
        return (img - m) / (s + eps)

    @staticmethod
    def _wrap_angle(dphi: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(dphi), torch.cos(dphi))

    @staticmethod
    def _diff_pad(x: torch.Tensor):
        d1 = torch.diff(x, prepend=x[:1])
        d2 = torch.diff(d1, prepend=d1[:1])
        return d1, d2

    # ---------- STFT family ----------
    @torch.no_grad()
    def _stft(self, x_1d: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x_1d,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._hann.to(x_1d.device),
            return_complex=True,
            center=True,
            normalized=False,
            onesided=True
        )  # [F,T] complex

    @torch.no_grad()
    def _build_logpower(self, spec: torch.Tensor) -> torch.Tensor:
        power = spec.real.pow(2) + spec.imag.pow(2)
        logp = torch.log(power + self.eps)
        if self.clip_db is not None:
            mx = logp.max()
            logp = torch.maximum(logp, mx - self.clip_db)
        ch = self._z(logp).unsqueeze(0).repeat(3, 1, 1)  # [3,F,T]
        return ch

    @torch.no_grad()
    def _build_stft_complex3(self, spec: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2))
        logmag = torch.log(mag + self.eps)
        if self.clip_db is not None:
            mx = logmag.max()
            logmag = torch.maximum(logmag, mx - self.clip_db)
        phase = torch.atan2(spec.imag, spec.real)  # [-pi, pi]
        cphi, sphi = torch.cos(phase), torch.sin(phase)
        ch0 = self._z(logmag)
        ch1 = self._z(cphi)
        ch2 = self._z(sphi)
        return torch.stack([ch0, ch1, ch2], dim=0)  # [3,F,T]

    # ---------- Recurrence plot ----------
    @torch.no_grad()
    def _recurrence3(self, x: torch.Tensor) -> torch.Tensor:
        xs = x[::self.rp_stride]
        d1, d2 = self._diff_pad(x)
        d1s = d1[::self.rp_stride]
        d2s = d2[::self.rp_stride]

        def rp(sig):
            v = self._z(sig)
            D = v.unsqueeze(1) - v.unsqueeze(0)  # [N,N]
            return torch.exp(-(D * D) / (2 * (self.rp_sigma ** 2) + 1e-12))

        R0 = self._z(rp(xs))
        R1 = self._z(rp(d1s))
        R2 = self._z(rp(d2s))
        return torch.stack([R0, R1, R2], dim=0)  # [3,N,N]

    # ---------- SpecAug ----------
    @staticmethod
    def _mask_axis(x: torch.Tensor, axis: int, max_width: int):
        # x: [C, F, T], axis=1 or 2
        if max_width <= 0: return x
        width = int(np.random.uniform(0, max_width + 1))
        if width == 0: return x
        t = x.size(axis)
        if t <= 1: return x
        start = int(np.random.uniform(0, max(1, t - width)))
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(start, start + width)
        x[tuple(sl)] = 0.0
        return x

    # ---------- main ----------
    def __getitem__(self, idx):
        x = torch.from_numpy(self._row(idx))  # [L]

        apply_specaug = False
        if self.rep == "logpower":
            spec = self._stft(x); img = self._build_logpower(spec); apply_specaug = True
        elif self.rep == "stft_complex3":
            spec = self._stft(x); img = self._build_stft_complex3(spec); apply_specaug = True
        elif self.rep == "recurrence3":
            img = self._recurrence3(x)
        else:
            raise RuntimeError("unreachable")

        # mask (all valid)
        mask = torch.ones(1, img.shape[1], img.shape[2], dtype=torch.float32)

        # SpecAug only for STFT-like reps
        if self.train_flag and self.spec_aug and apply_specaug:
            img = self._mask_axis(img, axis=1, max_width=self.freq_mask_param)
            img = self._mask_axis(img, axis=2, max_width=self.time_mask_param)

        # DINOv3/Imagenet normalization
        img = (img - self.img_mean) / self.img_std

        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, mask, y

# ---------------------------
# Collate (same as your DINOv3 code)
# ---------------------------
def collate_varlen(batch):
    imgs, masks, ys = zip(*batch)  # img: [3,F,T]
    T_max = max(im.shape[2] for im in imgs)
    imgs_p, masks_p = [], []
    for im, m in zip(imgs, masks):
        pad = T_max - im.shape[2]
        if pad > 0:
            im = F.pad(im, (0, pad), value=0.0)
            m  = F.pad(m,  (0, pad), value=0.0)
        imgs_p.append(im); masks_p.append(m)
    return torch.stack(imgs_p), torch.stack(masks_p), torch.tensor(ys, dtype=torch.float32)

# ---------------------------
# Dataset: PPG -> spectrogram image (3xF×T)
# ---------------------------
class NpySignalDataset2(Dataset):
    """
    Each .npy data file holds an array of shape [N, L] (mono PPG).
    Labels are diastolic BP in mmHg, shape [N].
    We build a log-power spectrogram and repeat it to 3 channels.

    New: robust filtering
      - drops rows where signal has NaN/Inf
      - optional label bounds (label_min/label_max)
      - optional custom label_valid_fn(y) -> bool mask
      - optional min_std to remove near-constant signals
      - chunked scanning to avoid loading whole X in RAM
      - optional keep_idx cache to skip re-scanning on later runs
    """
    def __init__(
        self,
        data_path: str,
        label_path: str,
        fs: int = 40,
        n_fft: int = 128,
        hop_length: int = 16,
        win_length: Optional[int] = None,
        clip_db: float = 80.0,
        eps: float = 1e-6,
        train: bool = False,
        spec_aug: bool = False,
        freq_mask_param: int = 4,
        time_mask_param: int = 4,
        # ---- filtering options ----
        label_min: Optional[float] = None,
        label_max: Optional[float] = None,
        label_valid_fn: Optional[callable] = None,  # fn(y: np.ndarray)-> np.ndarray[bool] of shape [N]
        min_std: float = 0.0,                        # drop if per-row std < min_std (0 disables)
        check_chunk: int = 10000,                   # rows per scan chunk
        keep_idx_cache: Optional[str] = None,       # path to .npy to cache keep indices
        verbose: bool = True,
    ):
        super().__init__()

        # Signals via mmap
        self.X = np.load(data_path, mmap_mode="r")      # [N, L]
        N, L = self.X.shape

        # Labels (small enough to load)
        y = np.load(label_path)
        if y.ndim > 1: y = y.squeeze()
        y = y.astype(np.float32, copy=False)
        if y.shape[0] != N:
            raise ValueError(f"Mismatched N between X ({N}) and y ({y.shape[0]})")

        # Try to load cached keep_idx
        keep_idx = None
        if keep_idx_cache and os.path.isfile(keep_idx_cache):
            try:
                cached = np.load(keep_idx_cache)
                if cached.ndim == 1 and cached.dtype.kind in "iu":
                    if cached.max(initial=-1) < N:
                        keep_idx = cached.astype(np.int64, copy=False)
                        if verbose:
                            print(f"[Dataset] Loaded keep_idx cache: {keep_idx_cache} (kept {keep_idx.size}/{N})")
            except Exception:
                keep_idx = None  # fall back to fresh scan

        if keep_idx is None:
            valid = np.ones(N, dtype=bool)

            # Label finite + optional bounds
            valid &= np.isfinite(y)
            if label_min is not None:
                valid &= (y >= label_min)
            if label_max is not None:
                valid &= (y <= label_max)

            if label_valid_fn is not None:
                extra = np.asarray(label_valid_fn(y), dtype=bool)
                if extra.shape != (N,):
                    raise ValueError("label_valid_fn must return boolean mask of shape [N].")
                valid &= extra

            # Scan X in chunks to drop NaN/Inf rows and (optionally) near-constant rows
            for i in range(0, N, check_chunk):
                j = min(N, i + check_chunk)
                Xi = np.asarray(self.X[i:j])  # view as ndarray (no copy on mmap)
                finite_rows = np.isfinite(Xi).all(axis=1)
                ok = finite_rows
                if min_std > 0.0:
                    ok &= (Xi.std(axis=1) >= float(min_std))
                valid[i:j] &= ok

            keep_idx = np.nonzero(valid)[0].astype(np.int64)

            # Optionally cache
            if keep_idx_cache:
                try:
                    os.makedirs(os.path.dirname(keep_idx_cache), exist_ok=True)
                except Exception:
                    pass
                try:
                    np.save(keep_idx_cache, keep_idx)
                    if verbose:
                        print(f"[Dataset] Saved keep_idx cache to {keep_idx_cache}")
                except Exception:
                    if verbose:
                        print(f"[Dataset] Warning: failed to save keep_idx cache to {keep_idx_cache}")

        self.keep_idx = keep_idx
        self.n = int(self.keep_idx.size)
        self.y = y[self.keep_idx].astype(np.float32, copy=False)

        # STFT params
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.clip_db = clip_db
        self.eps = eps

        # Augment flags
        self._hann = torch.hann_window(self.win_length)
        self.train_flag = train
        self.spec_aug = spec_aug
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

        # ImageNet-ish stats for ViTs (DINOv3)
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.img_std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        if verbose:
            dropped = N - self.n
            pct = 0.0 if N == 0 else 100.0 * dropped / N
            print(f"[Dataset] kept {self.n}/{N} ({100-pct:.2f}%), dropped {dropped} ({pct:.2f}%) "
                  f"(nan/inf/label/out-of-range/min_std)")

    def __len__(self):
        return self.n

    def _row(self, i: int) -> np.ndarray:
        # map new index -> original row
        return np.asarray(self.X[self.keep_idx[i]], dtype=np.float32)

    @torch.no_grad()
    def _spectrogram(self, x_1d: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x_1d,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._hann.to(x_1d.device),
            return_complex=True,
            center=True,
            normalized=False,
            onesided=True
        )  # [F, T]
        power = spec.real.pow(2) + spec.imag.pow(2)
        logp = torch.log(power + self.eps)

        # clamp dynamic range
        if self.clip_db is not None:
            mx = logp.max()
            logp = torch.maximum(logp, mx - self.clip_db)

        # per-sample z-norm
        logp = (logp - logp.mean()) / (logp.std() + 1e-6)

        img = logp.unsqueeze(0).repeat(3, 1, 1)  # [3, F, T]
        return img

    @staticmethod
    def _mask_axis(x: torch.Tensor, axis: int, max_width: int):
        # x: [C, F, T], axis=1 or 2
        if max_width <= 0: return x
        width = int(np.random.uniform(0, max_width + 1))
        if width == 0: return x
        t = x.size(axis)
        if t <= 1: return x
        start = int(np.random.uniform(0, max(1, t - width)))
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(start, start + width)
        x[tuple(sl)] = 0.0
        return x

    def __getitem__(self, idx):
        x = torch.from_numpy(self._row(idx))                # [L]
        img = self._spectrogram(x)                          # [3,F,T]
        mask = torch.ones(1, img.shape[1], img.shape[2], dtype=torch.float32)

        if self.train_flag and self.spec_aug:
            img = self._mask_axis(img, axis=1, max_width=self.freq_mask_param)
            img = self._mask_axis(img, axis=2, max_width=self.time_mask_param)

        # Normalize to ImageNet stats
        img = (img - self.img_mean) / self.img_std

        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, mask, y

# ---------------------------
# Mask-aware attention pooling over ViT token grid
# ---------------------------
class SoftAttentionPool2d(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.drop  = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor, mask_2d: torch.Tensor):
        # feats: [B,C,Hf,Wf], mask_2d: [B,1,Hf,Wf] (1=valid)
        logits = self.score(self.drop(feats))                 # [B,1,Hf,Wf]
        logits = logits.masked_fill(mask_2d < 0.5, -1e4)
        w = torch.softmax(logits.view(feats.size(0), 1, -1), dim=-1).view_as(logits)
        pooled = (feats * w).sum(dim=[2,3])                   # [B,C]
        return pooled

# ---------------------------
# LoRA targeting helper
# ---------------------------
def infer_lora_targets(
    model: nn.Module,
    explicit: Optional[List[str]] = None,
    include_mlp: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    Return a list of *exact* nn.Linear module names to adapt with LoRA.
    We only return leaf Linear names (no containers), to avoid PEFT trying
    to wrap whole attention blocks.
    """
    if explicit and explicit != ["auto"]:
        if verbose:
            print(f"[LoRA] Using explicit targets (verbatim): {explicit}")
        return explicit

    # Collect fully-qualified names for all Linear modules
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if verbose:
        print(f"[LoRA] Found {len(linear_names)} nn.Linear modules (sampling a few):")
        for n in linear_names[:8]:
            print("   -", n)

    # We only want attention projections (and optionally MLP)
    want_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if include_mlp:
        # matches your DINOv3 MLP naming
        want_suffixes += ["up_proj", "down_proj"]

    targets_exact = [n for n in linear_names if any(n.endswith(suf) for suf in want_suffixes)]

    if verbose:
        print(f"[LoRA] Exact Linear targets ({len(targets_exact)}):")
        for n in targets_exact[:12]:
            print("   -", n)
        if len(targets_exact) == 0:
            print("[LoRA] WARNING: No exact Linear targets were found with the expected suffixes. "
                  "Consider printing all Linear names and supplying --lora_targets with explicit names.")

    return targets_exact


# ---------------------------
# Frozen DINOv3 backbone + small MLP head (LoRA optional)
# ---------------------------
class Dinov3BPRegressor(nn.Module):
    """
    - Backbone: DINOv3 ViT-*/16 from HF
    - By default, freeze all backbone weights.
    - If use_lora=True, wrap attention (and optionally MLP) linears with LoRA adapters
      and train only LoRA params + pooling conv + head.
    - Mask-aware attention pooling over patch grid.
    - MLP head predicts diastolic BP (scalar).
    """
    def __init__(
        self,
        model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        head_hidden: int = 256,
        head_dropout: float = 0.1,
        local_files_only: bool = False,
        # LoRA args
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets: Optional[List[str]] = None,
        lora_on_mlp: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)
#        for p in self.backbone.parameters():
#            p.requires_grad = True
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        self.patch = int(getattr(self.backbone.config, "patch_size", 16))
        self.num_reg = int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.hidden = int(self.backbone.config.hidden_size)

        self.use_lora = use_lora and PEFT_AVAILABLE
        if use_lora and not PEFT_AVAILABLE:
            raise ImportError("peft not available. Install `peft` to use LoRA.")

        if self.use_lora:
            # if CLI gave strings and not "auto", treat them as *exact* names
            explicit = None
            if lora_targets and lora_targets != ["auto"]:
                explicit = lora_targets  # pass-through; expect exact names
            targets = infer_lora_targets(
                self.backbone,
                explicit=explicit,
                include_mlp=lora_on_mlp,
                verbose=True,
            )
            if not targets:
                raise ValueError("LoRA targeting found no Linear leaves. "
                                 "Try --lora_targets with explicit fully-qualified names.")
            peft_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=targets,     # <- exact Linear names
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, peft_cfg)
            # freeze non-LoRA
            for n, p in self.backbone.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.pool = SoftAttentionPool2d(self.hidden, dropout=head_dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x_img: torch.Tensor, mask_img: torch.Tensor) -> torch.Tensor:
        """
        x_img:   [B,3,F,T]
        mask_img:[B,1,F,T]
        """
        ps = self.patch
        B, _, F_in, T_in = x_img.shape
        # Resize to multiples of patch size
        Ht = max(ps, int(math.ceil(F_in / ps) * ps))
        Wt = max(ps, int(math.ceil(T_in / ps) * ps))
        if (F_in != Ht) or (T_in != Wt):
            x_img    = F.interpolate(x_img,    size=(Ht, Wt), mode="bilinear", align_corners=False)
            mask_img = F.interpolate(mask_img, size=(Ht, Wt), mode="nearest")

        out = self.backbone(pixel_values=x_img, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B, 1 + num_reg + N, C]
        patch_tokens = tokens[:, 1 + self.num_reg :, :]               # drop CLS + registers

        C = patch_tokens.size(-1)
        Hf, Wf = Ht // ps, Wt // ps
        feats_2d = patch_tokens.transpose(1, 2).reshape(B, C, Hf, Wf) # [B,C,Hf,Wf]
        mask_2d  = F.interpolate(mask_img, size=(Hf, Wf), mode="nearest")

        pooled = self.pool(feats_2d, mask_2d)                         # [B,C]
        pred   = self.head(pooled).squeeze(1)                         # [B]
        return pred

# ---------------------------
# Metrics / Eval
# ---------------------------
def mae(pred, target):  return torch.mean(torch.abs(pred - target)).item()
def rmse(pred, target): return torch.sqrt(torch.mean((pred - target)**2)).item()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, mb, yb in loader:
        xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
        out = model(xb, mb)
        preds.append(out.detach()); trues.append(yb.detach())
    preds = torch.cat(preds); trues = torch.cat(trues)
    return {"MAE": mae(preds, trues), "RMSE": rmse(preds, trues)}

# ---------------------------
# Train
# ---------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Datasets / Loaders
    train_ds = NpySignalDataset(
        args.train_data_path, args.train_label_path,
        fs=args.fs, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,
        train=True, spec_aug=args.spec_aug, freq_mask_param=args.freq_mask_param, time_mask_param=args.time_mask_param
    )
    val_ds   = NpySignalDataset(
        args.val_data_path,   args.val_label_path,
        fs=args.fs, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,
        train=False, spec_aug=False
    )
    test_ds  = NpySignalDataset(
        args.test_data_path,  args.test_label_path,
        fs=args.fs, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,
        train=False, spec_aug=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_varlen)
    val_loader   = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_varlen)
    test_loader  = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_varlen)

    # Model (frozen backbone or LoRA adapters)
    if args.lora_targets and args.lora_targets.lower() != "auto":
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    else:
        targets = ["auto"]

    model = Dinov3BPRegressor(
        model_id=args.model_name,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        local_files_only=args.local_files_only,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=targets,
        lora_on_mlp=args.lora_on_mlp,
        gradient_checkpointing=args.grad_checkpoint,
    ).to(device)

    # Collect only trainable params (LoRA + pool + head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Cosine schedule with warmup (epoch-stepped as in your original)
    total_steps = max(1, (len(train_loader) // max(1,args.grad_accum)) * args.epochs)
    warmup_steps = int(args.warmup * total_steps)
    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: cosine_with_warmup(s))

    scaler = torch.amp.GradScaler("cuda", enabled=(not args.no_amp) and torch.cuda.is_available())

    best_val = float("inf")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, mb, yb) in enumerate(train_loader, 1):
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            with torch.amp.autocast("cuda", enabled=(not args.no_amp) and torch.cuda.is_available()):
                pred = model(xb, mb)
                loss = F.smooth_l1_loss(pred, yb, beta=args.huber_beta)  # Huber (robust)
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = yb.numel()
            total_loss += float(loss.item()) * bs * args.grad_accum
            n_seen     += bs

            if step % args.log_every == 0:
                print(f"Epoch {epoch:03d}/{args.epochs} | Step {step:05d} | "
                      f"Loss {loss.item()*args.grad_accum:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                      f"Elapsed {human_time(time.time()-start)}", flush=True)

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        tr_loss = total_loss / max(1, n_seen)
        print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.4f} | "
              f"val_MAE={val_metrics['MAE']:.3f} | val_RMSE={val_metrics['RMSE']:.3f} | T={human_time(time.time()-start)}")

        # Save best by val MAE
        if val_metrics["MAE"] < best_val:
            best_val = val_metrics["MAE"]
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, "best.pt")
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "val": val_metrics}, save_path)
            print(f"  -> Saved best to {save_path}")

    # Final test with best
    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    test_metrics = evaluate(model, test_loader, device)
    print(json.dumps({"val_best_MAE": best_val, "test": test_metrics}, indent=2))

# ---------------------------
# CLI
# ---------------------------
def build_argparser():
    p = argparse.ArgumentParser("DINOv3 (frozen/LoRA) last-layer tuning for diastolic BP regression")
    # Paths (your defaults)
    p.add_argument("--train_data_path", type=str, default="ppg_bp/ppg_bp_diastolic/X_train_40.npy")
    p.add_argument("--train_label_path", type=str, default="ppg_bp/ppg_bp_diastolic/y_train.npy")
    p.add_argument("--val_data_path", type=str, default="ppg_bp/ppg_bp_diastolic/X_val_40.npy")
    p.add_argument("--val_label_path", type=str, default="ppg_bp/ppg_bp_diastolic/y_val.npy")
    p.add_argument("--test_data_path", type=str, default="ppg_bp/ppg_bp_diastolic/X_test_40.npy")
    p.add_argument("--test_label_path", type=str, default="ppg_bp/ppg_bp_diastolic/y_test.npy")

    # Signal -> Spectrogram
    p.add_argument("--fs", type=int, default=40)
    p.add_argument("--n_fft", type=int, default=128)
    p.add_argument("--hop_length", type=int, default=16)
    p.add_argument("--win_length", type=int, default=None)

    # Optional SpecAug
    p.add_argument("--spec_aug", action="store_true")
    p.add_argument("--freq_mask_param", type=int, default=4)
    p.add_argument("--time_mask_param", type=int, default=4)

    # Model / Head
    # p.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--model_name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
#    p.add_argument("--model_name", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
#    p.add_argument("--model_name", type=str, default="./dinov3_model")
    p.add_argument("--local_files_only", action="store_true") #, default=True)
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.1)

    # LoRA
    p.add_argument("--use_lora", default=True, action="store_true", help="Enable LoRA adapters on the backbone")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_on_mlp", action="store_true", help="Also apply LoRA to MLP fc1/fc2")
    p.add_argument("--lora_targets", type=str, default="auto",
                   help='Comma-separated target patterns or "auto" (qkv,proj[,fc1,fc2])')
    p.add_argument("--grad_checkpoint", action="store_true", help="Enable gradient checkpointing if supported")

    # Train
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)   # OK for head+LoRA
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.1)  # fraction of steps
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--huber_beta", type=float, default=2.0)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="./dinov3_bp_last_out3")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
