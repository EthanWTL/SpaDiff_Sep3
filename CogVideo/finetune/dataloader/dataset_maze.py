# dataset_maze.py
import os, json
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from CogVideo.finetune.dataloader.render import (
    # constants / geometry
    CANVAS_W, CANVAS_H, N_POS, R_DOT,
    OX, OY, CELL_W, CELL_H,
    # core flow/path + IO helpers (unchanged logic from your script)
    resize_fixed_zoom, paste_on_canvas_center,
    compute_centers_from_info, make_flow_frame,
    draw_grid_pil, draw_dot_pil,
    # HSV visualization
    flow_to_hsv_rgb_single,
    flow_to_hsv_channels_single,
)

class MazeFlowDataset(Dataset):
    """
    Returns per-item options:
      - flow video:  'flow'     -> [T, 2, H, W]    (u,v), float32
      - bg image:    'image'    -> [3, H, W]       (RGB 0..1), if return_image_bg=True
      - HSV video:   'flow_hsv' -> [T, 3, H, W]    (RGB 0..1), if return_hsv_seq=True
    T == seq_len (default 49).
    """

    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        seq_len: int = N_POS - 1,          # 49 flow frames
        zoom_s: Optional[float] = None,    # None => auto-calibrate from first image height
        to_tensor: bool = True,
        # --- options you asked for ---
        return_flow_seq: bool = True,
        return_image_bg: bool = False,
        return_hsv_seq: bool = False,
        return_hsv_channels_seq: bool = False,
        return_prompt: bool = True,        # Return prompt text
        # HSV settings
        hsv_clip_mag: Optional[float] = None,  # None => per-sample max
        hsv_saturation: float = 1.0,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.seq_len = int(seq_len)
        self.to_tensor = to_tensor

        self.return_flow_seq = bool(return_flow_seq)
        self.return_image_bg = bool(return_image_bg)
        self.return_hsv_seq = bool(return_hsv_seq)
        self.return_hsv_channels_seq = bool(return_hsv_channels_seq)
        self.return_prompt = bool(return_prompt)
        self.hsv_clip_mag = hsv_clip_mag
        self.hsv_saturation = float(hsv_saturation)

        with open(jsonl_path, "r") as f:
            self.records = [json.loads(line) for line in f]
        
        # Load prompts from the corresponding prompts.txt file
        self.prompts = []
        if self.return_prompt:
            prompts_path = os.path.join(os.path.dirname(jsonl_path), "prompts.txt")
            if os.path.exists(prompts_path):
                with open(prompts_path, "r") as f:
                    self.prompts = [line.strip() for line in f]
            else:
                print(f"Warning: prompts.txt not found at {prompts_path}")
                self.prompts = [""] * len(self.records)

        # Auto zoom from first image height to CANVAS_H
        if zoom_s is None:
            first_idx = self.records[0]["index"]
            p = os.path.join(self.images_dir, f"maze_{first_idx}.png")
            with Image.open(p) as im0:
                zoom_s = CANVAS_H / float(im0.size[1])
        self.zoom_s = float(zoom_s)

    def __len__(self) -> int:
        return len(self.records)

    # (optional) quicklook frame builder — unused for now but handy to keep
    def _make_quicklook_frame(self, bg_canvas: Image.Image, centers: List[tuple], t: int) -> Image.Image:
        img = bg_canvas.copy()
        d = ImageDraw.Draw(img)
        # full path
        if len(centers) >= 2:
            for i in range(len(centers) - 1):
                x0, y0 = centers[i]; x1, y1 = centers[i + 1]
                d.line([(x0, y0), (x1, y1)], fill=(0, 128, 0), width=3)
        # highlight segment t
        if 0 <= t < len(centers) - 1:
            x0, y0 = centers[t]; x1, y1 = centers[t + 1]
            d.line([(x0, y0), (x1, y1)], fill=(255, 165, 0), width=5)
            draw_dot_pil(img, x0, y0, R_DOT, (255, 0, 0))
            draw_dot_pil(img, x1, y1, R_DOT, (0, 255, 0))
        # grid
        draw_grid_pil(img, OX, OY, CELL_W, CELL_H)
        return img

    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = self.records[i]
        idx = rec["index"]
        img_path = os.path.join(self.images_dir, f"maze_{idx}.png")

        # Background on canvas
        orig = Image.open(img_path).convert("RGB")
        resized = resize_fixed_zoom(orig, self.zoom_s)
        bg_canvas, _, _ = paste_on_canvas_center(resized)

        # 50 centers -> 49 flows (exact logic from your script)
        centers = compute_centers_from_info(rec, N=N_POS)

        # Build flows_np: (T, H, W, 2)
        if not centers or len(centers) < 2:
            flows_np = np.zeros((self.seq_len, CANVAS_H, CANVAS_W, 2), dtype=np.float32)
        else:
            max_t = min(self.seq_len, len(centers) - 1)
            frames = [
                make_flow_frame(centers[t], centers[t + 1], H=CANVAS_H, W=CANVAS_W, r=R_DOT)
                for t in range(max_t)
            ]
            frames_np = np.stack(frames, axis=0) if frames else np.zeros((0, CANVAS_H, CANVAS_W, 2), dtype=np.float32)
            if max_t < self.seq_len:
                pad = np.zeros((self.seq_len - max_t, CANVAS_H, CANVAS_W, 2), dtype=np.float32)
                flows_np = np.concatenate([frames_np, pad], axis=0)
            else:
                flows_np = frames_np

        # Add third zero channel to make it compatible with CogVideoX VAE (needs 3 channels)
        # Convert from (T, H, W, 2) to (T, H, W, 3)
        zero_channel = np.zeros((flows_np.shape[0], flows_np.shape[1], flows_np.shape[2], 1), dtype=np.float32)
        flows_np_3ch = np.concatenate([flows_np, zero_channel], axis=3)  # (T, H, W, 3)

        out: Dict[str, Any] = {"index": idx}

        # flow video [T,3,H,W] - now with 3 channels for VAE compatibility
        if self.return_flow_seq:
            out["flow"] = torch.from_numpy(flows_np_3ch).permute(0, 3, 1, 2).contiguous() if self.to_tensor else flows_np_3ch

        # bg image [3,H,W] - normalized to [-1, 1] to match CogVideoX pipeline
        if self.return_image_bg:
            img_np = np.array(bg_canvas, dtype=np.uint8)  # (H,W,3)
            if self.to_tensor:
                # Apply same normalization as CogVideoX: /255.0 * 2.0 - 1.0 -> [-1, 1]
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
                out["image"] = (img_tensor / 255.0) * 2.0 - 1.0
            else:
                out["image"] = bg_canvas

        # HSV video [T,3,H,W] (RGB in 0..1) (Channels)
        # Decide magnitude clip for this sample (used by both RGB + channels)
        if self.hsv_clip_mag is None:
            mhi = float(np.sqrt((flows_np[..., 0] ** 2 + flows_np[..., 1] ** 2)).max())
            if mhi <= 1e-6: mhi = 1.0
        else:
            mhi = float(self.hsv_clip_mag)

        # HSV RGB video [T,3,H,W] (for display)
        if self.return_hsv_seq:
            hsv_rgb_list = [flow_to_hsv_rgb_single(flows_np[t], clip_mag=mhi, saturation=self.hsv_saturation)
                            for t in range(self.seq_len)]
            hsv_seq_rgb = np.stack(hsv_rgb_list, axis=0)  # (T,H,W,3) uint8
            out["flow_hsv"] = (torch.from_numpy(hsv_seq_rgb).permute(0,3,1,2).float() / 255.0) if self.to_tensor else hsv_seq_rgb

        # HSV CHANNELS video [T,3,H,W] (H,S,V in [0,1] float)
        if self.return_hsv_channels_seq:
            hsv_ch_list = [flow_to_hsv_channels_single(flows_np[t], clip_mag=mhi, saturation=self.hsv_saturation)
                           for t in range(self.seq_len)]
            hsv_seq_ch = np.stack(hsv_ch_list, axis=0).astype(np.float32)  # (T,H,W,3) float
            out["flow_hsv_channels"] = (torch.from_numpy(hsv_seq_ch).permute(0,3,1,2).contiguous()
                                        if self.to_tensor else hsv_seq_ch)

        if "labels" in rec:
            out["labels"] = rec["labels"]
        
        # Add prompt if requested
        if self.return_prompt:
            # Use the dataset position i (0-indexed) to match with prompts.txt lines
            if self.prompts and 0 <= i < len(self.prompts):
                out["prompt"] = self.prompts[i]
            else:
                out["prompt"] = ""  # fallback for missing prompts

        return out
