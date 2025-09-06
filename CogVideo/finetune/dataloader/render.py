# render.py
# -----------------------------------------------------------------------------
# Module version of your optical-flow generator.
# Math/logic for the core methods are kept EXACTLY the same as your script.
# Only minimal helpers are added at the bottom for convenient dataset usage.
# -----------------------------------------------------------------------------
import math
import numpy as np
from PIL import Image, ImageDraw
import numpy as np

# ========= MATCH GEOMETRY =========
CANVAS_W, CANVAS_H = 720, 480
ZOOM_S = 1.00

GRID_ROWS = 5
GRID_COLS = 5
OX, OY = 195, 33
CELL_W, CELL_H = 73, 73

# Exact pixel radius (must match draw script)
R_DOT = 16

# Number of OUTPUT positions = 50 -> 49 flow frames
N_POS = 50

# Visualization knobs (used by save_quicklook only)
VIS_T = 0
DRAW_FULL_PATH = True
DRAW_GRID = True
DRAW_DOTS = True  # draw circles with exact R_DOT


# ---------- ORIGINAL METHODS (unchanged logic) --------------------------------
def resize_fixed_zoom(img_pil, s):
    w0, h0 = img_pil.size
    w1, h1 = int(round(w0 * s)), int(round(h0 * s))
    return img_pil.resize((w1, h1), resample=Image.NEAREST)

def paste_on_canvas_center(img_resized, canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    w1, h1 = img_resized.size
    x0 = (canvas_w - w1) // 2
    y0 = (canvas_h - h1) // 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(img_resized, (x0, y0))
    return canvas, x0, y0

def cell_center_px(row, col, ox=OX, oy=OY, cw=CELL_W, ch=CELL_H):
    x = ox + col * cw + cw / 2.0
    y = oy + row * ch + ch / 2.0
    return float(x), float(y)

def cells_to_canvas_points(cell_path):
    return [cell_center_px(r, c) for (r, c) in cell_path]

def labels_to_canvas_points(start_rc, labels):
    r, c = start_rc
    pts = [cell_center_px(r, c)]
    for dir_frac, step_len in labels:
        f = (dir_frac % 1.0)
        if abs(f - 0.00) < 1e-6:   dr, dc = 0, +1
        elif abs(f - 0.25) < 1e-6: dr, dc = -1, 0
        elif abs(f - 0.50) < 1e-6: dr, dc = 0, -1
        elif abs(f - 0.75) < 1e-6: dr, dc = +1, 0
        else:
            theta = f * 2 * math.pi
            dr, dc = -math.sin(theta), math.cos(theta)
        steps = int(round(step_len / 0.2))
        for _ in range(steps):
            r += dr if isinstance(dr, int) else int(round(dr))
            c += dc if isinstance(dc, int) else int(round(dc))
            pts.append(cell_center_px(r, c))
    return pts

def resample_polyline(points, N=N_POS):
    if len(points) == 0: return []
    if len(points) == 1: return [points[0]] * N
    xs, ys = zip(*points)
    dists = [0.0]
    for i in range(1, len(points)):
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        dists.append(dists[-1] + math.hypot(dx, dy))
    total = dists[-1]
    if total == 0: return [points[0]] * N
    targets = [i * total / (N - 1) for i in range(N)]
    out, j = [], 0
    for t in targets:
        while j < len(dists) - 2 and dists[j+1] < t:
            j += 1
        t0, t1 = dists[j], dists[j+1]
        x0, y0 = points[j]
        x1, y1 = points[j+1]
        if t1 == t0:
            out.append((x0, y0))
        else:
            a = (t - t0) / (t1 - t0)
            out.append((x0 + a*(x1 - x0), y0 + a*(y1 - y0)))
    return out

def make_flow_frame(center_xy, next_xy, H=CANVAS_H, W=CANVAS_W, r=R_DOT):
    cx, cy = center_xy
    nx, ny = next_xy
    dx, dy = float(nx - cx), float(ny - cy)
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - cx)**2 + (yy - cy)**2 <= (r*r)
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[..., 0][mask] = dx
    flow[..., 1][mask] = dy
    return flow

def draw_grid_pil(img, ox, oy, cw, ch, rows=GRID_ROWS, cols=GRID_COLS):
    d = ImageDraw.Draw(img)
    for c in range(cols + 1):
        x = ox + c * cw
        d.line([(x, oy), (x, oy + rows * ch)], fill=(0,0,0), width=2)
    for r in range(rows + 1):
        y = oy + r * ch
        d.line([(ox, y), (ox + cols * cw, y)], fill=(0,0,0), width=2)

def draw_dot_pil(img, x, y, r, color):
    d = ImageDraw.Draw(img)
    d.ellipse([(x-r, y-r), (x+r, y+r)], fill=color)

def save_quicklook(bg_canvas, centers, t, out_png, ox, oy, cw, ch):
    img = bg_canvas.copy()
    d = ImageDraw.Draw(img)
    if DRAW_GRID:
        draw_grid_pil(img, ox, oy, cw, ch)
    if DRAW_FULL_PATH and len(centers) >= 2:
        for i in range(len(centers)-1):
            x0, y0 = centers[i]
            x1, y1 = centers[i+1]
            d.line([(x0, y0), (x1, y1)], fill=(0,128,0), width=3)
    if 0 <= t < len(centers)-1:
        x0, y0 = centers[t]
        x1, y1 = centers[t+1]
        d.line([(x0, y0), (x1, y1)], fill=(255,165,0), width=5)
        if DRAW_DOTS:
            draw_dot_pil(img, x0, y0, R_DOT, (255,0,0))
            draw_dot_pil(img, x1, y1, R_DOT, (0,255,0))
    img.save(out_png)


# ---------- SMALL HELPERS (composition only; no logic changes) ----------------
def compute_centers_from_info(info, N=N_POS):
    """
    Build the resampled 50 centers (=> 49 flow frames) from an 'info' record.
    Uses EXACTLY the same path construction rules as your script.
    """
    if "true_path" in info and info["true_path"]:
        pts = cells_to_canvas_points(info["true_path"])
    elif "labels" in info and info["labels"]:
        lf = info["labels"]
        labels_pairs = lf[1] if (isinstance(lf, list) and len(lf) == 2 and isinstance(lf[1], list)) else lf
        pts = labels_to_canvas_points(info["start_pos"], labels_pairs)
    else:
        return []  # caller can decide what to do
    return resample_polyline(pts, N=N)

def make_flow_sequence_from_info(info, H=CANVAS_H, W=CANVAS_W, r=R_DOT, N=N_POS):
    """
    Produce a numpy array of shape (N-1, H, W, 3) with 49 flow frames,
    using the EXACT same make_flow_frame logic around a disk of radius R_DOT.
    The 3rd channel is filled with zeros (for VAE compatibility).
    """
    centers = compute_centers_from_info(info, N=N)
    if len(centers) < 2:
        return np.zeros((max(N-1, 1), H, W, 3), dtype=np.float32)
    
    frames = [make_flow_frame(centers[t], centers[t+1], H=H, W=W, r=r) for t in range(N-1)]
    flow = np.stack(frames, axis=0)  # (N-1, H, W, 2)

    # Add an extra all-zero channel
    zeros = np.zeros(flow.shape[:-1] + (1,), dtype=np.float32)  # (N-1, H, W, 1)
    flow = np.concatenate([flow, zeros], axis=-1)  # (N-1, H, W, 3)

    return flow

def _hsv_to_rgb_np(hsv):  # hsv in [0,1], returns rgb in [0,1]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)

def flow_to_hsv_rgb_single(flow_hw2: np.ndarray, clip_mag: float | None = None, saturation: float = 1.0) -> np.ndarray:
    """
    flow_hw2: (H, W, 2) float32, (u,v) with v down-positive (image coords)
    Returns uint8 RGB image mapping angle->hue, mag->value, sat=saturation (inside mask).
    Outside the motion disk (mag==0), S=0 so result is grayscale black (V=0).
    """
    assert flow_hw2.ndim == 3 and flow_hw2.shape[-1] == 2, f"Expected (H,W,2), got {flow_hw2.shape}"
    u, v = flow_hw2[..., 0], flow_hw2[..., 1]
    mag = np.sqrt(u * u + v * v)
    ang = (np.arctan2(v, u) + np.pi) / (2 * np.pi)  # [0,1)
    if clip_mag is None:
        mhi = float(mag.max()) if mag.size else 1.0
    else:
        mhi = max(clip_mag, 1e-6)
    val = np.clip(mag / (mhi + 1e-6), 0.0, 1.0)

    # Saturation only where mag>0; elsewhere set S=0 so background stays neutral
    s = np.zeros_like(val)
    s[mag > 0] = float(saturation)

    hsv = np.stack([ang, s, val], axis=-1)
    rgb01 = _hsv_to_rgb_np(hsv)
    return (np.clip(rgb01 * 255.0, 0, 255)).astype(np.uint8)

def flow_to_hsv_channels_single(flow_hw2: np.ndarray, clip_mag: float | None = None, saturation: float = 1.0) -> np.ndarray:
    """
    flow_hw2: (H,W,2) float32 (u,v); v is down-positive.
    Returns float32 HSV channels in [0,1] as (H,W,3) = (H, S, V):
      - H = angle / (2π) in [0,1)
      - S = saturation inside disk (mag>0), else 0
      - V = mag normalized by clip_mag (or per-frame max if None)
    """
    assert flow_hw2.ndim == 3 and flow_hw2.shape[-1] == 2, f"Expected (H,W,2), got {flow_hw2.shape}"
    u, v = flow_hw2[..., 0], flow_hw2[..., 1]
    mag = np.sqrt(u*u + v*v)
    ang01 = (np.arctan2(v, u) + np.pi) / (2 * np.pi)  # [0,1)
    if clip_mag is None:
        mhi = float(mag.max()) if mag.size else 1.0
        if mhi <= 1e-6: mhi = 1.0
    else:
        mhi = max(float(clip_mag), 1e-6)
    val = np.clip(mag / mhi, 0.0, 1.0)
    sat = np.zeros_like(val, dtype=np.float32)
    sat[mag > 0] = float(saturation)
    hsv = np.stack([ang01.astype(np.float32), sat, val.astype(np.float32)], axis=-1)  # (H,W,3)
    return hsv

    
__all__ = [
    # constants
    "CANVAS_W", "CANVAS_H", "ZOOM_S",
    "GRID_ROWS", "GRID_COLS", "OX", "OY", "CELL_W", "CELL_H",
    "R_DOT", "N_POS", "VIS_T", "DRAW_FULL_PATH", "DRAW_GRID", "DRAW_DOTS",
    # original methods
    "resize_fixed_zoom", "paste_on_canvas_center", "cell_center_px",
    "cells_to_canvas_points", "labels_to_canvas_points", "resample_polyline",
    "make_flow_frame", "draw_grid_pil", "draw_dot_pil", "save_quicklook",
    # helpers
    "compute_centers_from_info", "make_flow_sequence_from_info",
]
