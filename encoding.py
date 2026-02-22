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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ─────────────────────────────────────────────
#  EO ENCODER
#  EfficientNet-B0 pretrained on ImageNet
#  Input:  [B, 3, 224, 224]
#  Output: [B, 1280]
# ─────────────────────────────────────────────

class EOEncoder(nn.Module):
    """
    EO (Electro-Optical / RGB) encoder.

    Uses EfficientNet-B0 pretrained on ImageNet. The final classifier
    layer is removed — we only want the 1280-dim feature vector from
    the adaptive average pool, not class predictions.

    Why EfficientNet over ResNet-18?
        EfficientNet scales depth, width, and resolution together using
        a compound coefficient. B0 is the smallest variant but still
        outperforms ResNet-18 on ImageNet with fewer parameters.
        Better features = better downstream fusion performance.

    Args:
        freeze_backbone: If True, freeze all EfficientNet weights and
                         only train the projection head. Useful when
                         your dataset is small.
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()

        # Load pretrained EfficientNet-B0
        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # EfficientNet architecture:
        #   effnet.features   → convolutional feature extractor
        #   effnet.avgpool    → adaptive average pool → [B, 1280, 1, 1]
        #   effnet.classifier → Linear(1280, 1000)  ← we remove this
        self.features  = effnet.features
        self.avgpool   = effnet.avgpool
        self.feature_dim = 1280

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EO image tensor [B, 3, 224, 224]
        Returns:
            features: [B, 1280]
        """
        x = self.features(x)     # [B, 1280, 7, 7]
        x = self.avgpool(x)      # [B, 1280, 1, 1]
        x = x.flatten(1)         # [B, 1280]
        return x


# ─────────────────────────────────────────────
#  SAR ENCODER
# ##############NOT TRAINED FROM SCRATCH ANYMORE. PRETRAINED EFFICIENTNET-B0 WITH ADAPTED 1-CHANNEL INPUT CONV
# ─────────────────────────────────────────────

class SAREncoder(nn.Module):
    """
    SAR (Synthetic Aperture Radar) encoder.

    Uses EfficientNet-B0 with ImageNet pretrained weights adapted for 
    1-channel input by averaging the RGB conv weights → grayscale.
    
    Why pretrained → adapted?
        Training from scratch is slow and risky with limited data.
        ImageNet pretrained weights capture useful low-level features
        (edges, textures) that transfer well to SAR. Averaging the
        3-channel conv weights into 1 channel preserves these features
        while adapting to grayscale input.
    
    How we adapt for 1-channel input:
        1. Load pretrained EfficientNet-B0 (3-channel Conv2d)
        2. Average RGB weights: [32, 3, 3, 3] → [32, 1, 3, 3]
        3. Replace first conv with adapted 1-channel conv
    """

    def __init__(self, use_pretrained_init: bool = True):
        super().__init__()

        # Load EfficientNet-B0 WITH pretrained weights for transfer
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained_init else None
        effnet = efficientnet_b0(weights=weights)

        # The first conv layer is nested inside features[0][0]
        original_conv = effnet.features[0][0]

        # Create new 1-channel conv layer
        new_conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = original_conv.out_channels,
            kernel_size  = original_conv.kernel_size,
            stride       = original_conv.stride,
            padding      = original_conv.padding,
            bias         = False
        )
        
        # Initialize from pretrained: average RGB weights → grayscale
        if use_pretrained_init:
            with torch.no_grad():
                # original_conv.weight shape: [32, 3, 3, 3] → mean over dim=1 → [32, 1, 3, 3]
                new_conv.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        effnet.features[0][0] = new_conv

        self.features    = effnet.features
        self.avgpool     = effnet.avgpool
        self.feature_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: SAR image tensor [B, 1, 224, 224]
        Returns:
            features: [B, 1280]
        """
        x = self.features(x)     # [B, 1280, 7, 7]
        x = self.avgpool(x)      # [B, 1280, 1, 1]
        x = x.flatten(1)         # [B, 1280]
        return x


# ─────────────────────────────────────────────
#  COMBINED ENCODER
# ─────────────────────────────────────────────

class ModalityEncoders(nn.Module):
    """
    Wraps both encoders into a single module for clean forward passes.

    Args:
        freeze_eo_backbone: freeze EO EfficientNet weights (default False)
    """

    def __init__(self, freeze_eo_backbone: bool = False):
        super().__init__()
        self.eo_encoder  = EOEncoder(freeze_backbone=freeze_eo_backbone)
        self.sar_encoder = SAREncoder()

    def forward(self, eo: torch.Tensor, sar: torch.Tensor):
        """
        Args:
            eo  : [B, 3, 224, 224]
            sar : [B, 1, 224, 224]
        Returns:
            eo_feat  : [B, 1280]
            sar_feat : [B, 1280]
        """
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