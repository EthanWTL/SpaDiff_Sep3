from CogVideo.finetune.dataloader.loader import build_all_loaders
from CogVideo.finetune.dataloader.flow_norm import scan_flow_stats, merge_stats, save_stats, load_stats, make_normalizing_collate
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader


train = "/project/sds-rise/ethan/SpaDiff_Sep3/datasets/maze/10000maze_720_480/info_labels.jsonl"
train_img = "/project/sds-rise/ethan/SpaDiff_Sep3/datasets/maze/10000maze_720_480/images"
val   = "/project/sds-rise/ethan/SpaDiff_Sep3/datasets/maze/validation/info_labels.jsonl"
val_img = "/project/sds-rise/ethan/SpaDiff_Sep3/datasets/maze/validation/images"
test  = "/project/sds-rise/ethan/SpaDiff_Sep3/datasets/maze/evaluation/info_labels.jsonl"
test_img = "/project/sds-rise/ethan/SpaDiff_Sep3/datasets/maze/evaluation/images"


# Build loaders as-is (no normalization yet)
(train_ds, train_dl), (val_ds, val_dl), (test_ds, test_dl) = build_all_loaders(
    train_jsonl=train, train_images=train_img,
    val_jsonl=val,     val_images=val_img,
    test_jsonl=test,   test_images=test_img,
    batch_size=1,
    num_workers=0,
    seq_len=49,
    return_flow_seq=False,
    return_image_bg=True,
    return_hsv_seq=True,
    return_hsv_channels_seq=False,
    return_prompt=True,  # Enable prompt loading for testing
    hsv_clip_mag=None,
    hsv_saturation=1.0,
)

"""
# Scan stats (use quantile=0.999 for 99.9th percentile, or None for hard max)
print("Begin scaning the validation records")
val_stats   = scan_flow_stats(val_dl,   key="flow", quantile=0.999)
print("Done scanning the validation records")
print("Begin scaning the test records")
test_stats  = scan_flow_stats(test_dl,  key="flow", quantile=0.999)
print("Done scanning the test records")
print("Begin scaning the training records")
train_stats = scan_flow_stats(train_dl, key="flow", quantile=0.999)
print("Done scanning the training records")

# Merge: max across all splits
merged_stats = merge_stats([train_stats, val_stats, test_stats])

save_stats("./scripts/dataloader_exp/flow_norm_stats_10000maze.json", merged_stats)

scale_mag = merged_stats["max_mag"]


scale_mag = load_stats("./dataloader_exp/flow_norm_stats_10000maze.json")["max_mag"]# Wrap a collate_fn that normalizes + clips
norm_collate = make_normalizing_collate(default_collate, scale_mag, key="flow", mode="mag", clip=True)

"""

# Rebuild normalized loaders
train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False)
test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=False)

# Check
batch = next(iter(train_dl))
print("Flow HSV max after norm+clip:", batch["flow_hsv"].abs().max().item())  # <= 1.0
print("HSV shape:", batch["flow_hsv"].shape)  # Should be [B, T, 3, H, W]
print("Index:", batch["index"])
print("Prompt:", batch["prompt"])

# Check image normalization
if "image" in batch:
    img_min = batch["image"].min().item()
    img_max = batch["image"].max().item()
    print(f"Image range after normalization: [{img_min:.3f}, {img_max:.3f}] (should be ~[-1, 1])")
else:
    print("No image found in batch")

breakpoint()






