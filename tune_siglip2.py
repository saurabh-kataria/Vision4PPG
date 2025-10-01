#!/usr/bin/env python3
# lo/train_siglip2_bp_lora.py
import os, math, time, argparse, json
from typing import Optional, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel
from transformers import SiglipVisionModel

# --- PEFT / LoRA ---
from peft import LoraConfig, get_peft_model, TaskType
# pick a vision-friendly task type


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

# ---------------------------
# Dataset: PPG -> log-power spectrogram image (3 x F x T)
# (No SpecAug; normalized to SigLIP-style [-1,1] by default)
# ---------------------------

class NpySignalDataset(Dataset):
    """
    Robust dataset for PPG -> spectrogram:
    - Filters rows with NaN/Inf in signal
    - Filters labels that are NaN/Inf or out of [label_min, label_max]
    - Optional: drop nearly-constant signals (min_std)
    - Uses mmap and chunked scanning for big arrays
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
        channels: int = 3,
        siglip_norm: bool = True,
        # ---- filtering options ----
        label_min: Optional[float] = None,
        label_max: Optional[float] = None,
        label_valid_fn: Optional[callable] = None,  # fn(y: np.ndarray)-> np.ndarray[bool]
        min_std: float = 0.0,         # drop signals with std < min_std (0 = disabled)
        check_chunk: int = 10000,     # rows per chunk when scanning X
        verbose: bool = True,
    ):
        super().__init__()
        # mmap signals
        self.X = np.load(data_path, mmap_mode="r")  # [N, L]
        # labels (materialize—they're small)
        y = np.load(label_path)
        if y.ndim > 1: y = y.squeeze()
        y = y.astype(np.float32, copy=False)

        if self.X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched N between X ({self.X.shape[0]}) and y ({y.shape[0]})")

        # ---- build keep_idx mask ----
        N, L = self.X.shape
        valid = np.ones(N, dtype=bool)

        # label finite and optional bounds
        label_finite = np.isfinite(y)
        valid &= label_finite
        if label_min is not None:
            valid &= (y >= label_min)
        if label_max is not None:
            valid &= (y <= label_max)

        if label_valid_fn is not None:
            try:
                extra_mask = np.asarray(label_valid_fn(y), dtype=bool)
                if extra_mask.shape != (N,):
                    raise ValueError("label_valid_fn must return a boolean mask of shape [N].")
                valid &= extra_mask
            except Exception as e:
                raise RuntimeError(f"label_valid_fn failed: {e}")

        # scan X by chunks to find non-finite rows (and optional low-std rows)
        for i in range(0, N, check_chunk):
            j = min(N, i + check_chunk)
            Xi = np.asarray(self.X[i:j])  # (chunk, L)
            finite_rows = np.isfinite(Xi).all(axis=1)
            ok = finite_rows
            if min_std > 0.0:
                ok &= (Xi.std(axis=1) >= float(min_std))
            valid[i:j] &= ok

        self.keep_idx = np.nonzero(valid)[0].astype(np.int64)
        self.n = int(self.keep_idx.size)

        # store labels for kept rows only
        self.y = y[self.keep_idx].astype(np.float32, copy=False)

        # spectrogram params
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.clip_db = clip_db
        self.eps = eps
        self.channels = channels
        self._hann = torch.hann_window(self.win_length)

        # normalization for SigLIP ([-1,1])
        if siglip_norm:
            mean = [0.5, 0.5, 0.5]
            std  = [0.5, 0.5, 0.5]
        else:
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
        self.img_mean = torch.tensor(mean).view(3,1,1)
        self.img_std  = torch.tensor(std).view(3,1,1)

        if verbose:
            dropped = N - self.n
            pct = 0.0 if N == 0 else 100.0 * dropped / N
            print(f"[Dataset] kept {self.n}/{N} ({100-pct:.2f}%), dropped {dropped} ({pct:.2f}%) "
                  f"(nan/inf/label/out-of-range/min_std)")

    def __len__(self):
        return self.n

    def _row(self, i: int) -> np.ndarray:
        # map filtered index -> original row in memmap
        return np.asarray(self.X[self.keep_idx[i]], dtype=np.float32)

    @torch.no_grad()
    def _spectrogram(self, x_1d: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x_1d, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self._hann.to(x_1d.device), return_complex=True, center=True,
            normalized=False, onesided=True
        )
        power = spec.real.pow(2) + spec.imag.pow(2)
        logp = torch.log(power + self.eps)
        if self.clip_db is not None:
            mx = logp.max()
            logp = torch.maximum(logp, mx - self.clip_db)
        logp = (logp - logp.mean()) / (logp.std() + 1e-6)
        img = logp.unsqueeze(0).repeat(3, 1, 1)  # [3,F,T]
        return img

    def __getitem__(self, idx):
        x = torch.from_numpy(self._row(idx))              # [L] float32
        img = self._spectrogram(x)                        # [3,F,T]
        mask = torch.ones(1, img.shape[1], img.shape[2], dtype=torch.float32)
        img = (img - self.img_mean) / self.img_std        # SigLIP norm
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, mask, y


# ---------------------------
# Collate: right-pad time dimension
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
# LoRA target utilities
# ---------------------------
def parse_lora_targets_arg(arg: Optional[str]) -> Optional[List[str]]:
    if arg is None: return None
    arg = arg.strip()
    if not arg: return None
    if arg.lower() == "auto": return ["auto"]
    return [t.strip() for t in arg.split(",") if t.strip()]

def infer_lora_targets(
    model: nn.Module,
    explicit: Optional[List[str]] = None,
    include_mlp: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    Return a list of *exact* nn.Linear module names to adapt with LoRA.
    For SigLIP(2) ViT towers, Linear leaves typically end with:
    q_proj, k_proj, v_proj, out_proj (attention)
    MLP: fc1, fc2
    """
    if explicit and explicit != ["auto"]:
        if verbose:
            print(f"[LoRA] Using explicit targets: {explicit}")
        return explicit

    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if verbose:
        print(f"[LoRA] Found {len(linear_names)} nn.Linear modules (sampling a few):")
        for n in linear_names[:10]:
            print("   -", n)

    want_suffixes = ["q_proj", "k_proj", "v_proj", "out_proj"]
    if include_mlp:
        want_suffixes += ["fc1", "fc2", "mlp.fc1", "mlp.fc2"]

    targets_exact = [n for n in linear_names if any(n.endswith(suf) for suf in want_suffixes)]

    if verbose:
        print(f"[LoRA] Exact Linear targets ({len(targets_exact)}):")
        for n in targets_exact[:16]:
            print("   -", n)
        if len(targets_exact) == 0:
            print("[LoRA] WARNING: no matches. Provide --lora_targets with exact names printed above.")
    return targets_exact

# replace your VisionForwardWrapper with this
class VisionForwardWrapper(torch.nn.Module):
    """
    Let PEFT's FEATURE_EXTRACTION wrapper call a vision model safely.
    Maps input_ids/inputs_embeds -> pixel_values and drops text-only kwargs.
    """
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.model = base_model  # SiglipVisionModel

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        inputs_embeds=None,
        interpolate_pos_encoding: bool = True,   # <-- default to True
        **kwargs
    ):
        if pixel_values is None:
            if input_ids is not None:
                pixel_values = input_ids
            elif inputs_embeds is not None:
                pixel_values = inputs_embeds
            else:
                raise TypeError("Need pixel_values or (input_ids/inputs_embeds).")

        # keep only kwargs SiglipVisionModel.forward accepts
        allowed = {"output_attentions", "output_hidden_states", "return_dict", "interpolate_pos_encoding"}
        call_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        call_kwargs.setdefault("interpolate_pos_encoding", True)  # <-- ensure True

        return self.model(pixel_values=pixel_values, **call_kwargs)

# ---------------------------
# SigLIP-2 regressor with optional LoRA
# ---------------------------

class SigLIP2BPRegressor(nn.Module):
    """
    SigLIP-2 vision tower + mask-aware attention pooling + MLP head.
    LoRA (optional) is applied ONLY to the vision tower.
    """
    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch14-384",
        head_hidden: int = 256,
        head_dropout: float = 0.1,
        local_files_only: bool = False,
        # LoRA
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets: Optional[List[str]] = None,  # exact names or ["auto"]
        lora_on_mlp: bool = False,
        gradient_checkpointing: bool = False,
        cache_dir: Optional[str] = None,           # optional if you added --cache_dir
    ):
        super().__init__()

        # 1) Load the vision tower (not the dual/text+vision model)
        base_vision = SiglipVisionModel.from_pretrained(
            model_id, local_files_only=local_files_only,
            cache_dir=getattr(self, "cache_dir", None) if hasattr(self, "cache_dir") else None,
        )

        if gradient_checkpointing and hasattr(base_vision, "gradient_checkpointing_enable"):
            base_vision.gradient_checkpointing_enable()

        # Read config from the *vision* tower (has hidden_size/patch_size)
        vcfg = base_vision.config
        self.patch  = int(getattr(vcfg, "patch_size", 14))
        self.hidden = int(getattr(vcfg, "hidden_size"))
        self.num_reg = int(getattr(vcfg, "num_register_tokens", 0))  # usually 0

        # 2) Wrap the vision model so PEFT can pass input_ids safely
        self.vision = VisionForwardWrapper(base_vision)

        # 3) LoRA on the wrapper (so it reaches inside the vision tower)
        self.use_lora = use_lora
        if self.use_lora:
            targets = infer_lora_targets(
                self.vision, explicit=lora_targets, include_mlp=lora_on_mlp, verbose=True
            )
            if not targets:
                raise ValueError("LoRA targeting found no Linear layers in the vision tower.")
            peft_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", target_modules=targets,
                task_type=TaskType.FEATURE_EXTRACTION,  # keep this for older PEFT
            )
            self.vision = get_peft_model(self.vision, peft_cfg)

            # Freeze non-LoRA params
            for n, p in self.vision.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False
        else:
            for p in self.vision.parameters():
                p.requires_grad = False

        # --------- Pool + head ----------
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

        # Resize to multiples of patch
        Ht = max(ps, int(math.ceil(F_in / ps) * ps))
        Wt = max(ps, int(math.ceil(T_in / ps) * ps))
        if (F_in != Ht) or (T_in != Wt):
            x_img    = F.interpolate(x_img,    size=(Ht, Wt), mode="bilinear", align_corners=False)
            mask_img = F.interpolate(mask_img, size=(Ht, Wt), mode="nearest")

        target_dtype = next(self.vision.parameters()).dtype
        x_img = x_img.to(dtype=target_dtype)

        # IMPORTANT: call the vision tower directly to get token grid
        out = self.vision(
            pixel_values=x_img,
            output_hidden_states=False,
            interpolate_pos_encoding=True,   # <-- key line
        )
        tokens = out.last_hidden_state

        patch_tokens = tokens[:, 1 + self.num_reg :, :]  # drop CLS(+registers) if any

        C = patch_tokens.size(-1)
        ps = self.patch
        Hf, Wf = Ht // ps, Wt // ps
        expected = Hf * Wf + self.num_reg
        n_tok = tokens.size(1)

        # Detect if a CLS token exists; SigLIP vision usually does NOT have one.
        if n_tok == expected:
            cls = 0
        elif n_tok == expected + 1:
            cls = 1
        else:
            # Fallback: assume no CLS; you can also raise if this happens repeatedly
            cls = max(0, n_tok - expected)

        patch_tokens = tokens[:, cls + self.num_reg :, :]  # [B, Npatch, C]
        feats_2d = patch_tokens.transpose(1, 2).reshape(B, C, Hf, Wf)
        mask_2d  = F.interpolate(mask_img, size=(Hf, Wf), mode="nearest")

        pooled = self.pool(feats_2d, mask_2d)            # [B, C]
        pred   = self.head(pooled).squeeze(1)            # [B]
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
    )
    val_ds   = NpySignalDataset(
        args.val_data_path,   args.val_label_path,
        fs=args.fs, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,
    )
    test_ds  = NpySignalDataset(
        args.test_data_path,  args.test_label_path,
        fs=args.fs, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,
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

    # Parse lora_targets CLI
    lt = parse_lora_targets_arg(args.lora_targets)

    # Model
    model = SigLIP2BPRegressor(
        model_id=args.model_name,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        local_files_only=args.local_files_only,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=lt,
        lora_on_mlp=args.lora_on_mlp,
        gradient_checkpointing=args.grad_checkpoint,
    ).to(device)

    # Collect trainable params (LoRA + pool + head)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n.lower() for nd in ["bias", "ln", "layernorm", "norm"]):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )

    # Cosine schedule with warmup
    total_steps = max(1, (len(train_loader) // max(1, args.grad_accum)) * args.epochs)
    warmup_steps = int(args.warmup * total_steps)
    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: cosine_with_warmup(s))

    scaler = torch.amp.GradScaler("cuda", enabled=(not args.no_amp) and torch.cuda.is_available())

    # Print trainable counts
    tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params] Trainable: {tot:,}")

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
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
    p = argparse.ArgumentParser("SigLIP-2 (LoRA optional) for diastolic BP regression from PPG spectrograms")
    # Paths (edit to yours)
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

    # Model / Head
#    p.add_argument("--model_name", type=str, default="google/siglip2-so400m-patch14-384")
    p.add_argument("--model_name", type=str, default="google/siglip2-large-patch16-384")
#    p.add_argument("--model_name", type=str, default="google/siglip2-g-patch16-384")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.1)

    # LoRA
    p.add_argument("--use_lora", default=True, action="store_true", help="Enable LoRA adapters on the backbone")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_on_mlp", default=True, action="store_true", help="Also apply LoRA to MLP fc1/fc2")
    p.add_argument("--lora_targets", type=str, default="auto",
                   help='Exact Linear names (comma-separated) or "auto"')

    # Train
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--lr", type=float, default=5e-4)   # OK for head + LoRA
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--huber_beta", type=float, default=2.0)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="./siglip2_bp_lora_out")

    p.add_argument("--grad_checkpoint", action="store_true", help="Enable gradient checkpointing if supported")
    p.add_argument("--cache_dir", type=str, default="./cache", help="HuggingFace cache directory (models/tokenizers/configs)")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
~
