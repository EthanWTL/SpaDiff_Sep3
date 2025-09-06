# loader.py
from torch.utils.data import DataLoader
from CogVideo.finetune.dataloader.dataset_maze import MazeFlowDataset
import numpy as np, random, torch

def worker_init_fn(worker_id: int):
    # Per-worker determinism
    base_seed = 42
    seed = base_seed + worker_id
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

def _make_generator(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def build_loader(
    jsonl: str,
    images: str,
    batch_size: int = 1,
    num_workers: int = 1,
    pin_memory: bool = True,
    seq_len: int = 49,
    # --- three knobs + channels ---
    return_flow_seq: bool = True,          # adds batch['flow']     -> [B,T,2,H,W]
    return_image_bg: bool = False,         # adds batch['image']    -> [B,3,H,W]
    return_hsv_seq: bool = False,          # adds batch['flow_hsv'] -> [B,T,3,H,W] (RGB vis)
    return_hsv_channels_seq: bool = False, # adds batch['flow_hsv_channels'] -> [B,T,3,H,W] (H,S,V)
    return_prompt: bool = True,            # adds batch['prompt']   -> [B] (text strings)
    # HSV controls
    hsv_clip_mag = None,
    hsv_saturation: float = 1.0,
    # loader behavior
    shuffle: bool = True,
    seed: int = 42,
):
    ds = MazeFlowDataset(
        jsonl_path=jsonl,
        images_dir=images,
        seq_len=seq_len,
        zoom_s=None,
        to_tensor=True,
        return_flow_seq=return_flow_seq,
        return_image_bg=return_image_bg,
        return_hsv_seq=return_hsv_seq,
        return_hsv_channels_seq=return_hsv_channels_seq,
        return_prompt=return_prompt,
        hsv_clip_mag=hsv_clip_mag,
        hsv_saturation=hsv_saturation,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=_make_generator(seed),
    )
    return ds, dl

# --------------------------
# Convenience constructors
# --------------------------
def build_train_loader(
    jsonl: str,
    images: str,
    **kwargs
):
    # Train: shuffle on, use seed for determinism
    kwargs.setdefault("shuffle", True)
    kwargs.setdefault("seed", 42)
    return build_loader(jsonl=jsonl, images=images, **kwargs)

def build_val_loader(
    jsonl: str,
    images: str,
    **kwargs
):
    # Val: shuffle off
    kwargs.setdefault("shuffle", False)
    kwargs.setdefault("seed", 1337)
    return build_loader(jsonl=jsonl, images=images, **kwargs)

def build_test_loader(
    jsonl: str,
    images: str,
    **kwargs
):
    # Test: shuffle off
    kwargs.setdefault("shuffle", False)
    kwargs.setdefault("seed", 2024)
    return build_loader(jsonl=jsonl, images=images, **kwargs)

def build_all_loaders(
    train_jsonl: str, train_images: str,
    val_jsonl: str,   val_images: str,
    test_jsonl: str = None, test_images: str = None,
    **common_kwargs
):
    """
    Build train/val(/test) with shared dataset options (seq_len, returns, hsv settings, etc.).
    You can still override per-split by passing explicit kwargs to the convenience functions.
    """
    ds_train, dl_train = build_train_loader(train_jsonl, train_images, **common_kwargs)
    ds_val,   dl_val   = build_val_loader(val_jsonl,   val_images,   **common_kwargs)

    ds_test = dl_test = None
    if test_jsonl is not None and test_images is not None:
        ds_test, dl_test = build_test_loader(test_jsonl, test_images, **common_kwargs)

    return (ds_train, dl_train), (ds_val, dl_val), (ds_test, dl_test)
