"""
====================================================
MAVIC-C | Step 7: OOD Detection + Validation Evaluation
====================================================
Loads the best trained model and runs inference on the
validation set (SAR-only).

For each sample:
  - Predicts class label
  - Computes energy score → low = in-distribution, high = OOD
  - Compares against OOD_flag ground truth

Metrics reported:
  Classification : Accuracy, Macro-F1 (in-distribution only)
  OOD Detection  : AUROC, AUPR, FPR@95%TPR
====================================================
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from preprocessing import SAROnlyDataset, sar_val_transform
from torch.utils.data import DataLoader
from encoding import ModalityEncoders
from fusionModel import FusionModel


# ─────────────────────────────────────────────
#  CONFIG — update paths to match your setup
# ─────────────────────────────────────────────

CONFIG = {
    "checkpoint_path": "D:\\RWoodzell Classification Challenge\\checkpoints\\best_model.pth",
    "val_sar_root":    "D:\\RWoodzell Classification Challenge\\val",
    "val_csv_path":    "D:\\RWoodzell Classification Challenge\\val\\validation_reference.csv",
    "batch_size":      256,
    "num_workers":     16,
}


# ─────────────────────────────────────────────
#  MAVIC MODEL (same as train.py)
# ─────────────────────────────────────────────

class MAVICModel(nn.Module):
    def __init__(self, num_classes=10, drop_prob=0.25, label_smooth=0.1):
        super().__init__()
        self.encoders     = ModalityEncoders(freeze_eo_backbone=False)
        self.fusion_model = FusionModel(
            num_classes  = num_classes,
            drop_prob    = drop_prob,
            label_smooth = label_smooth,
        )

    def forward(self, eo: torch.Tensor, sar: torch.Tensor):
        eo_feat, sar_feat = self.encoders(eo, sar)
        return self.fusion_model(eo_feat, sar_feat)


# ─────────────────────────────────────────────
#  ENERGY SCORE
# ─────────────────────────────────────────────

def compute_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute energy score from logits.

    Formula: E(x) = -T * log( sum( exp(logits / T) ) )

    Properties:
        - In-distribution samples  → LOW energy  (confident predictions)
        - OOD samples              → HIGH energy  (uncertain predictions)

    The temperature T controls the smoothness:
        - T=1.0  : standard energy score (good default)
        - T>1.0  : softer scores, better separation in some cases
        - Try T=1.0 first, tune if AUROC is low

    Args:
        logits      : raw class scores [B, num_classes]
        temperature : scaling factor (default 1.0)
    Returns:
        energy : [B] — one score per sample
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


# ─────────────────────────────────────────────
#  FPR @ 95% TPR
# ─────────────────────────────────────────────

def fpr_at_95_tpr(ood_labels: np.ndarray, energy_scores: np.ndarray) -> float:
    """
    Compute False Positive Rate when True Positive Rate = 95%.

    In OOD detection:
        TPR = correctly flagging OOD samples as OOD
        FPR = incorrectly flagging in-distribution samples as OOD

    We want: high TPR (catch OOD) with low FPR (don't flag clean samples).
    FPR@95%TPR is the standard benchmark — lower is better.

    Args:
        ood_labels    : ground truth (1=OOD, 0=in-dist) [N]
        energy_scores : energy score per sample [N]
                        higher energy = more likely OOD
    Returns:
        fpr : float between 0 and 1
    """
    # Sort thresholds from low to high energy
    thresholds = np.sort(energy_scores)

    best_fpr = 1.0
    for thresh in thresholds:
        # Predict OOD if energy > threshold
        predicted_ood = (energy_scores >= thresh).astype(int)

        tp = np.sum((predicted_ood == 1) & (ood_labels == 1))
        fn = np.sum((predicted_ood == 0) & (ood_labels == 1))
        fp = np.sum((predicted_ood == 1) & (ood_labels == 0))
        tn = np.sum((predicted_ood == 0) & (ood_labels == 0))

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)

        if tpr >= 0.95:
            best_fpr = fpr
            break

    return best_fpr


# ─────────────────────────────────────────────
#  MAIN EVALUATION
# ─────────────────────────────────────────────

def evaluate(config: dict):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Load Checkpoint ---
    print(f"Loading checkpoint: {config['checkpoint_path']}")
    checkpoint    = torch.load(config["checkpoint_path"], map_location=device)
    class_to_idx  = checkpoint["class_to_idx"]
    num_classes   = len(class_to_idx)
    idx_to_class  = {v: k for k, v in class_to_idx.items()}

    print(f"Classes ({num_classes}): {class_to_idx}")
    print(f"Best val acc during training: {checkpoint['val_acc']*100:.2f}%\n")

    # --- Load Model ---
    model = MAVICModel(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()   # ← critical: disables modality dropout and batchnorm train mode

    # --- Validation Dataset ---
    val_dataset = SAROnlyDataset(
        sar_root      = config["val_sar_root"],
        csv_path      = config["val_csv_path"],
        class_to_idx  = class_to_idx,
        sar_transform = sar_val_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = config["num_workers"],
        pin_memory  = True,
    )

    # --- Inference ---
    print("Running inference...")
    all_energy      = []   # energy score per sample
    all_ood_flags   = []   # ground truth OOD label (0=in-dist, 1=OOD)
    all_preds       = []   # predicted class index
    all_true_labels = []   # true class index (-1 for OOD)

    with torch.no_grad():
        for batch in val_loader:
            eo       = batch['eo'].to(device,    non_blocking=True)  # zeros
            sar      = batch['sar'].to(device,    non_blocking=True)
            labels   = batch['label']
            ood_flag = batch['ood_flag']

            logits = model(eo, sar)                        # [B, num_classes]
            energy = compute_energy(logits, temperature=1.0)  # [B]
            preds  = logits.argmax(dim=1)                  # [B]

            all_energy.extend(energy.cpu().numpy())
            all_ood_flags.extend(ood_flag.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.numpy())

    all_energy      = np.array(all_energy)
    all_ood_flags   = np.array(all_ood_flags)
    all_preds       = np.array(all_preds)
    all_true_labels = np.array(all_true_labels)

    # ─────────────────────────────────────────
    #  CLASSIFICATION METRICS (in-dist only)
    # ─────────────────────────────────────────

    in_dist_mask    = (all_ood_flags == 0)
    in_dist_preds   = all_preds[in_dist_mask]
    in_dist_labels  = all_true_labels[in_dist_mask]

    acc      = accuracy_score(in_dist_labels, in_dist_preds)
    macro_f1 = f1_score(in_dist_labels, in_dist_preds, average='macro', zero_division=0)

    # ─────────────────────────────────────────
    #  OOD DETECTION METRICS
    # ─────────────────────────────────────────
    # Higher energy = more likely OOD
    # AUROC/AUPR expect: higher score = more likely positive (OOD)
    # so we pass energy scores directly as the OOD score

    auroc      = roc_auc_score(all_ood_flags, all_energy)
    aupr       = average_precision_score(all_ood_flags, all_energy)
    fpr95      = fpr_at_95_tpr(all_ood_flags, all_energy)

    # ─────────────────────────────────────────
    #  PRINT RESULTS
    # ─────────────────────────────────────────

    print("\n" + "=" * 50)
    print("CLASSIFICATION (in-distribution samples only)")
    print("=" * 50)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Macro-F1  : {macro_f1*100:.2f}%")

    print("\n" + "=" * 50)
    print("OOD DETECTION")
    print("=" * 50)
    print(f"  AUROC        : {auroc*100:.2f}%  (higher is better, 50% = random)")
    print(f"  AUPR         : {aupr*100:.2f}%  (higher is better)")
    print(f"  FPR@95%TPR   : {fpr95*100:.2f}%  (lower is better, 0% = perfect)")

    # ─────────────────────────────────────────
    #  ENERGY DISTRIBUTION SUMMARY
    # ─────────────────────────────────────────

    in_dist_energy = all_energy[in_dist_mask]
    ood_energy     = all_energy[~in_dist_mask]

    print("\n" + "=" * 50)
    print("ENERGY SCORE DISTRIBUTION")
    print("=" * 50)
    print(f"  In-distribution : mean={in_dist_energy.mean():.3f}, "
          f"std={in_dist_energy.std():.3f}")
    print(f"  OOD             : mean={ood_energy.mean():.3f}, "
          f"std={ood_energy.std():.3f}")
    print(f"  Separation gap  : {ood_energy.mean() - in_dist_energy.mean():.3f} "
          f"(positive = OOD has higher energy ✅)")

    # ─────────────────────────────────────────
    #  SAVE RESULTS TO CSV
    # ─────────────────────────────────────────

    results_df = pd.DataFrame({
        "predicted_class": [idx_to_class.get(p, "unknown") for p in all_preds],
        "true_class":      [idx_to_class.get(l, "OOD") for l in all_true_labels],
        "energy_score":    all_energy,
        "ood_flag_true":   all_ood_flags,
        "ood_flag_pred":   (all_energy > np.median(all_energy)).astype(int),
    })

    out_path = config["checkpoint_path"].replace("best_model.pth", "val_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to: {out_path}")
    print("\n✅ Step 7 complete.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    evaluate(CONFIG)