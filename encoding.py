"""
====================================================
MAVIC-C | Step 2: Modality-Specific Encoders
====================================================
EO  → EfficientNet-B0 pretrained on ImageNet (3-channel input)
SAR → EfficientNet-B0 trained from scratch   (1-channel input)

Both encoders strip the final classifier and output a
1280-dimensional feature vector per image.

NOTE: Feature dim is 1280 (not 512 like ResNet-18).
      Update FeatureProjector(in_dim=1280) in steps3to6_fusion_model.py
====================================================
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
)

# SARCLIP INTEGRATION
import sys
import logging
sys.path.insert(0, r'D:\RWoodzell Classification Challenge\BestModelTryAgain\SARCLIP')
import open_clip
from safetensors.torch import load_file


# Supported backbones and their feature dimensions
BACKBONE_REGISTRY = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
    "efficientnet_b1": (efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1, 1280),
    "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1, 1408),
    "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1, 1536),
}


def _build_efficientnet(backbone_name: str):
    """Load pretrained EfficientNet, strip classifier, return (features, avgpool, feat_dim)."""
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{backbone_name}'. Choose from: {list(BACKBONE_REGISTRY)}")
    model_fn, weights, feat_dim = BACKBONE_REGISTRY[backbone_name]
    net = model_fn(weights=weights)
    return net.features, net.avgpool, feat_dim


# ─────────────────────────────────────────────
#  EO ENCODER
#  EfficientNet-B0 pretrained on ImageNet
#  Input:  [B, 3, 224, 224]
#  Output: [B, 1280]
# ─────────────────────────────────────────────

class EOEncoder(nn.Module):
    """
    EO (Electro-Optical / RGB) encoder.
    Supports EfficientNet-B0 through B3 via the backbone argument.
    """

    def __init__(self, freeze_backbone: bool = False, backbone: str = "efficientnet_b0"):
        super().__init__()
        self.features, self.avgpool, self.feature_dim = _build_efficientnet(backbone)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x


# ─────────────────────────────────────────────
#  SAR ENCODER
# ##############NOT TRAINED FROM SCRATCH ANYMORE. PRETRAINED EFFICIENTNET-B0 WITH ADAPTED 1-CHANNEL INPUT CONV
# ─────────────────────────────────────────────

# LEGACY: replaced by SARCLIPEncoder
class SAREncoder(nn.Module):
    """
    SAR encoder — pretrained EfficientNet adapted for 1-channel input.
    Supports B0 through B3 via the backbone argument.
    """

    def __init__(self, use_pretrained_init: bool = True, backbone: str = "efficientnet_b0"):
        super().__init__()

        if backbone not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(BACKBONE_REGISTRY)}")
        model_fn, weights, feat_dim = BACKBONE_REGISTRY[backbone]
        effnet = model_fn(weights=weights if use_pretrained_init else None)

        # Adapt first conv layer from 3-channel → 1-channel
        original_conv = effnet.features[0][0]
        new_conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = original_conv.out_channels,
            kernel_size  = original_conv.kernel_size,
            stride       = original_conv.stride,
            padding      = original_conv.padding,
            bias         = False,
        )
        if use_pretrained_init:
            with torch.no_grad():
                new_conv.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        effnet.features[0][0] = new_conv

        self.features    = effnet.features
        self.avgpool     = effnet.avgpool
        self.feature_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x


# ─────────────────────────────────────────────
#  SARCLIP INTEGRATION: ViT-L-14 SAR Encoder
# ─────────────────────────────────────────────

class SARCLIPEncoder(nn.Module):
    """
    SAR encoder using pretrained SARCLIP ViT-L-14 weights.
    Loads HuggingFace .safetensors, remaps keys to open_clip format,
    and extracts the visual backbone.

    Input:  [B, 1, 224, 224]  (single-channel SAR)
    Output: [B, 768]          (ViT-L-14 embedding)
    """

    def __init__(self, weights_path: str, freeze_backbone: bool = True):
        super().__init__()

        # Create ViT-L-14 architecture — suppress the "no pretrained weights" log from open_clip
        _root_logger = logging.getLogger()
        _prev_level  = _root_logger.level
        _root_logger.setLevel(logging.ERROR)
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)
        _root_logger.setLevel(_prev_level)

        # Load safetensors weights — keys are open_clip format under vision_model.* prefix
        state_dict = load_file(weights_path)
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('vision_model.'):
                remapped[k.replace('vision_model.', 'visual.')] = v
            elif k == 'logit_scale':
                remapped[k] = v

        # strict=False: text tower keys are absent (intentional — we only use visual)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        # Verify all visual weights loaded (only text tower should be missing)
        visual_missing = [k for k in missing if k.startswith('visual.')]
        if visual_missing:
            raise RuntimeError(f"SARCLIP visual weights failed to load: {visual_missing[:5]}")
        print(f"[OK] SARCLIP ViT-L-14 loaded - visual keys: {len(remapped)-1}, text missing (expected): {len(missing)}")

        self.backbone    = model.visual  # ViT-L-14 visual encoder
        self.feature_dim = 768           # ViT-L-14 outputs 768-dim features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[FROZEN] SARCLIP backbone frozen")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand to 3 channels only if input is single-channel
        # (SARCLIP transforms already expand to 3ch; legacy transforms don't)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        return x   # [B, 768]


# ─────────────────────────────────────────────
#  COMBINED ENCODER
# ─────────────────────────────────────────────

class ModalityEncoders(nn.Module):
    """
    Wraps both encoders into a single module for clean forward passes.

    Args:
        freeze_eo_backbone : freeze EO backbone weights (default False)
        backbone           : which EfficientNet variant to use for both encoders
                             Options: 'efficientnet_b0' (default), 'b1', 'b2', 'b3'
    """

    def __init__(
        self,
        freeze_eo_backbone:  bool = False,
        backbone:            str  = "efficientnet_b0",
        # SARCLIP INTEGRATION
        freeze_sar_backbone: bool = True,
        sar_weights_path:    str  = '',
    ):
        super().__init__()
        self.eo_encoder  = EOEncoder(freeze_backbone=freeze_eo_backbone, backbone=backbone)

        # SARCLIP INTEGRATION: use SARCLIPEncoder instead of EfficientNet SAREncoder
        self.sar_encoder = SARCLIPEncoder(
            weights_path     = sar_weights_path,
            freeze_backbone  = freeze_sar_backbone,
        )

        self.feature_dim     = self.eo_encoder.feature_dim   # legacy compat
        self.eo_feature_dim  = self.eo_encoder.feature_dim   # 1280
        self.sar_feature_dim = self.sar_encoder.feature_dim  # 768

    def forward(self, eo: torch.Tensor, sar: torch.Tensor):
        eo_feat  = self.eo_encoder(eo)
        sar_feat = self.sar_encoder(sar)
        return eo_feat, sar_feat


# ─────────────────────────────────────────────
#  SANITY CHECK
# ─────────────────────────────────────────────
'''
if __name__ == "__main__":
    print("=" * 55)
    print("Step 2: EfficientNet Encoder Sanity Check")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    dummy_eo  = torch.randn(4, 3, 224, 224).to(device)
    dummy_sar = torch.randn(4, 1, 224, 224).to(device)

    eo_enc  = EOEncoder(freeze_backbone=False).to(device)
    sar_enc = SAREncoder().to(device)

    with torch.no_grad():
        eo_feat  = eo_enc(dummy_eo)
        sar_feat = sar_enc(dummy_sar)

    print(f"EO  encoder output : {eo_feat.shape}")    # [4, 1280]
    print(f"SAR encoder output : {sar_feat.shape}")   # [4, 1280]

    encoders = ModalityEncoders().to(device)
    with torch.no_grad():
        eo_f, sar_f = encoders(dummy_eo, dummy_sar)

    print(f"\nCombined encoder:")
    print(f"  EO  features : {eo_f.shape}")           # [4, 1280]
    print(f"  SAR features : {sar_f.shape}")           # [4, 1280]

    eo_params  = sum(p.numel() for p in eo_enc.parameters())
    sar_params = sum(p.numel() for p in sar_enc.parameters())
    print(f"\nEO  encoder params : {eo_params:,}")
    print(f"SAR encoder params : {sar_params:,}")

    print("\n⚠️  IMPORTANT: Update FeatureProjector(in_dim=1280)")
    print("   in steps3to6_fusion_model.py — was 512, now 1280")
    print("\n✅ Step 2 complete. Ready for Step 3: Feature Projection.")'''