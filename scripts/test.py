import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "/project/sds-rise/ethan/SpaDiff_Sep3/datasets/maze/10000maze_720_480/images/maze_24.png"
image = load_image(url)

pretrained_model_name = "/project/sds-rise/ethan/SpaDiff_Sep3/huggingface/models/dinov3-vit7b16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name, do_resize=False,do_center_crop=False,)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state
patch_tokens = last_hidden_state[:, 5:, :]  # (1, 1350, 4096)

# Step 2: reshape into grid (batch, height, width, hidden_dim)
batch_size, num_patches, hidden_dim = patch_tokens.shape
h, w = 30, 45  # from 480/16, 720/16
patch_grid = patch_tokens.reshape(batch_size, h, w, hidden_dim)
breakpoint()