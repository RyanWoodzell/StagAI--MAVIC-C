import torch
import open_clip
from safetensors.torch import load_file

weights_path = r'D:\RWoodzell Classification Challenge\BestModelTryAgain\StagAI--MAVIC-C\sar_clip\model_configs\ViT-L-14\models--BiliSakura--SARCLIP-ViT-L-14\snapshots\fd6c03457e79e65285acf0045f63ce6bc485650f\model.safetensors'

# Load weights
state_dict = load_file(weights_path)

# Remap vision_model.xxx → visual.xxx (only keep visual keys)
remapped = {}
for k, v in state_dict.items():
    if k.startswith('vision_model.'):
        new_key = k.replace('vision_model.', 'visual.')
        remapped[new_key] = v
    elif k == 'logit_scale':
        remapped[k] = v  # keep this too

print(f"Remapped {len(remapped)} keys")

# Create model
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name = 'ViT-L-14',
    pretrained = None,
)

# Load remapped weights
missing, unexpected = model.load_state_dict(remapped, strict=False)
print(f"Missing keys:    {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

# Missing keys should now only be text encoder keys (expected - we don't need those)
# Unexpected keys should be 0 or very low
print("\nUnexpected keys (should be empty):")
for k in unexpected:
    print(k)

# Test visual encoder
image_encoder = model.visual
dummy = torch.randn(2, 1, 224, 224)
dummy_3ch = dummy.repeat(1, 3, 1, 1)

with torch.no_grad():
    features = image_encoder(dummy_3ch)

print(f"\nOutput shape: {features.shape}")  # [2, 768]
print("✅ Visual encoder loaded with SAR pretrained weights!")