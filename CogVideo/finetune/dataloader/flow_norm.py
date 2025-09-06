# flow_norm.py
import json
from typing import Dict, Optional, Tuple
import torch
from tqdm import tqdm

@torch.no_grad()
def scan_flow_stats(
    dataloader,
    key: str = "flow",
    device: Optional[torch.device] = None,
    quantile: Optional[float] = None,
    max_samples: int = 200_000,   # cap number of samples
) -> Dict[str, float]:
    """
    Compute dataset-wide flow stats.
    If quantile=None: use hard maxima.
    If quantile: estimate by subsampling or per-batch quantile.
    """
    max_abs_u, max_abs_v, max_mag = 0.0, 0.0, 0.0
    samples = []

    for batch in tqdm(dataloader, desc="Scanning flow stats"):
        flow = batch[key]  # [B,T,2,H,W]
        if device is not None:
            flow = flow.to(device, non_blocking=True)

        u, v = flow[:, :, 0], flow[:, :, 1]
        max_abs_u = max(max_abs_u, float(u.abs().amax().item()))
        max_abs_v = max(max_abs_v, float(v.abs().amax().item()))

        mag = torch.linalg.vector_norm(flow, dim=2)  # [B,T,H,W]

        if quantile is None:
            max_mag = max(max_mag, float(mag.amax().item()))
        else:
            # collect a subsample
            flat = mag.reshape(-1)
            if flat.numel() > 10_000:
                idx = torch.randint(0, flat.numel(), (10_000,), device=flat.device)
                flat = flat[idx]
            samples.append(flat.cpu())

    if quantile is not None and samples:
        # merge and subsample again if too big
        cat = torch.cat(samples)
        if cat.numel() > max_samples:
            idx = torch.randint(0, cat.numel(), (max_samples,))
            cat = cat[idx]
        max_mag = float(torch.quantile(cat, quantile).item())

    eps = 1e-6
    return {
        "max_abs_u": max(max_abs_u, eps),
        "max_abs_v": max(max_abs_v, eps),
        "max_mag":   max(max_mag,   eps),
    }
    
def merge_stats(stats_list: list[Dict[str, float]]) -> Dict[str, float]:
    """Take the max across splits."""
    merged = {"max_abs_u": 0.0, "max_abs_v": 0.0, "max_mag": 0.0}
    for st in stats_list:
        merged["max_abs_u"] = max(merged["max_abs_u"], st["max_abs_u"])
        merged["max_abs_v"] = max(merged["max_abs_v"], st["max_abs_v"])
        merged["max_mag"]   = max(merged["max_mag"],   st["max_mag"])
    return merged

def save_stats(path: str, stats: Dict) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

def load_stats(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def normalize_flow_in_place(
    batch: Dict,
    scale,
    key: str = "flow",
    mode: str = "mag",
    clip: bool = True,
) -> Dict:
    """
    Normalize and optionally clip to [-1,1].
    """
    flow = batch[key]
    if mode == "mag":
        s = float(scale)
        batch[key] = (flow / s).clamp_(-1.0, 1.0) if clip else (flow / s)
    elif mode == "per_component":
        su, sv = scale
        flow[:, :, 0].div_(float(su))
        flow[:, :, 1].div_(float(sv))
        if clip:
            flow.clamp_(-1.0, 1.0)
    else:
        raise ValueError("mode must be 'mag' or 'per_component'")
    return batch

def make_normalizing_collate(default_collate, scale, key: str = "flow", mode: str = "mag", clip: bool = True):
    def _collate(items):
        batch = default_collate(items)
        return normalize_flow_in_place(batch, scale, key=key, mode=mode, clip=clip)
    return _collate
