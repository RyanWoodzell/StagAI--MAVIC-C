"""
====================================================
MAVIC-C | Steps 3–6: Projection → Fusion → Dropout → Classification
====================================================
Step 3: Project EO [B,1280] and SAR [B,1280] → shared [B,256] each
Step 4: Concatenate → fused [B,512]
Step 5: Modality Dropout — randomly zero one modality during training
Step 6: Classification head [B,512] → [B,num_classes]
====================================================
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  STEP 3: FEATURE PROJECTION
# ─────────────────────────────────────────────

class ModalityProjector(nn.Module):
    """
    Projects a single modality's features into a common feature space.
    Each modality gets its own projector with independent weights.
    """
    # BUG FIXED: in_dim was 512 — must be 1280 to match EfficientNet-B0 output
    def __init__(self, in_dim: int = 1280, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        '''
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        '''
        ############### UPDATES
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),           # GELU outperforms ReLU in transformer-adjacent models
            nn.Dropout(p=dropout),
            nn.Linear(768, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)   # [B, 256]


# ─────────────────────────────────────────────
#  STEP 5: MODALITY DROPOUT
# ─────────────────────────────────────────────

# BUG FIXED: ModalityDropout was missing the class definition entirely.
# __init__ and forward were loose functions — they would crash on import.
class ModalityDropout(nn.Module):
    """
    Randomly zeros out one entire modality's embedding during training.
    Forces the model to learn from either modality alone, which is
    critical since validation is SAR-only with no paired EO images.

    drop EO  probability = drop_prob / 2
    drop SAR probability = drop_prob / 2
    keep both            = 1 - drop_prob
    """
    def __init__(self, drop_prob: float = 0.25):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, eo_proj: torch.Tensor, sar_proj: torch.Tensor):
        if not self.training or self.drop_prob == 0.0:
            return eo_proj, sar_proj

        B      = eo_proj.size(0)
        half_p = self.drop_prob / 2.0
        r      = torch.rand(B, device=eo_proj.device)

        # BUG FIXED: drop_eo_mask was missing .float().unsqueeze(1)
        # A raw bool mask multiplied against a float tensor gives wrong results
        drop_eo_mask  = (r < half_p).float().unsqueeze(1)
        drop_sar_mask = ((r >= half_p) & (r < self.drop_prob)).float().unsqueeze(1)

        eo_proj  = eo_proj  * (1.0 - drop_eo_mask)
        sar_proj = sar_proj * (1.0 - drop_sar_mask)

        return eo_proj, sar_proj


# ─────────────────────────────────────────────
#  STEP 4: LATE FUSION
# ─────────────────────────────────────────────

class LateFusion(nn.Module):
    """
    Concatenates EO and SAR projected embeddings into a single vector.
    [B, 256] + [B, 256] → [B, 512]
    """
    def __init__(self):
        super().__init__()

    def forward(self, eo_proj: torch.Tensor, sar_proj: torch.Tensor) -> torch.Tensor:
        return torch.cat([eo_proj, sar_proj], dim=1)


# ─────────────────────────────────────────────
#  STEP 6: CLASSIFICATION HEAD
# ─────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Maps fused 512-dim representation to class logits.
    Architecture: Linear(512→256) → ReLU → Dropout → Linear(256→num_classes)
    """
    def __init__(
        self,
        in_dim:       int   = 512,
        num_classes:  int   = 10,
        dropout:      float = 0.4,
        label_smooth: float = 0.1,
    ):
        super().__init__()

        # BUG FIXED: second Linear was hardcoded as Linear(256, num_classes)
        # Using mid_dim consistently avoids a shape mismatch crash
        mid_dim = in_dim // 2

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mid_dim, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, labels)


# ─────────────────────────────────────────────
#  FULL MODEL (Steps 3–6 combined)
# ─────────────────────────────────────────────

class FusionModel(nn.Module):
    """
    Combines Steps 3–6 into one forward pass.

    Flow:
        eo_feat, sar_feat  [B, 1280]   (from EfficientNet encoders)
             ↓ Step 3
        eo_proj [B,256], sar_proj [B,256]
             ↓ Step 5 (training only)
        modality dropout
             ↓ Step 4
        fused [B, 512]
             ↓ Step 6
        logits [B, num_classes]
    """

    def __init__(
        self,
        num_classes:  int   = 10,
        proj_dim:     int   = 256,
        drop_prob:    float = 0.25,
        head_dropout: float = 0.4,
        label_smooth: float = 0.1,
        eo_in_dim:    int   = 1280,
        sar_in_dim:   int   = 1280,
    ):
        super().__init__()

        # Step 3: projection
        # BUG FIXED: was ModalityProjector(in_dim=512) — must be 1280 for EfficientNet
        self.eo_projector  = ModalityProjector(in_dim=eo_in_dim,  out_dim=proj_dim)
        self.sar_projector = ModalityProjector(in_dim=sar_in_dim, out_dim=proj_dim)

        # Step 5: modality dropout
        self.modality_dropout = ModalityDropout(drop_prob=drop_prob)

        # Step 4: late fusion
        self.fusion = LateFusion()

        # Step 6: classification head
        self.head = ClassificationHead(
            in_dim      = proj_dim * 2,
            num_classes = num_classes,
            dropout     = head_dropout,
            label_smooth= label_smooth,
        )

    def forward(self, eo_feat: torch.Tensor, sar_feat: torch.Tensor):
        eo_proj  = self.eo_projector(eo_feat)
        sar_proj = self.sar_projector(sar_feat)
        eo_proj, sar_proj = self.modality_dropout(eo_proj, sar_proj)
        fused  = self.fusion(eo_proj, sar_proj)
        logits = self.head(fused)
        return logits

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.head.compute_loss(logits, labels)

'''
# ─────────────────────────────────────────────
#  SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Steps 3–6: Fusion Model Sanity Check")
    print("=" * 55)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    batch_size  = 4

    dummy_eo_feat  = torch.randn(batch_size, 1280).to(device)
    dummy_sar_feat = torch.randn(batch_size, 1280).to(device)
    dummy_labels   = torch.randint(0, num_classes, (batch_size,)).to(device)

    model = FusionModel(num_classes=num_classes).to(device)

    model.train()
    logits = model(dummy_eo_feat, dummy_sar_feat)
    loss   = model.compute_loss(logits, dummy_labels)
    print(f"[Train] Logits : {logits.shape}")
    print(f"[Train] Loss   : {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(dummy_eo_feat, dummy_sar_feat)
    print(f"[Eval]  Logits : {logits.shape}")

    print("\n✅ Steps 3–6 complete. Ready for Step 7: OOD Detection.")
    '''