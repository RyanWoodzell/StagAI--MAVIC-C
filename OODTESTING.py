"""
====================================================
MAVIC-C | OOD Method Comparison
====================================================
Tests 4 OOD scoring methods on the validation set
and reports AUROC, AUPR, and TNR@TPR95 for each.

Methods compared:
  1. Energy Score       : -logsumexp(logits)
  2. Softmax Confidence : max(softmax(logits))
  3. Maximum Logit Score: max(logits)  — no softmax
  4. Entropy            : -sum(p * log(p))

TNR@TPR95 = 1 - FPR@95%TPR (what the competition scores)
Higher TNR@TPR95 = better OOD detection
====================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

from preprocessing import SAROnlyDataset, sar_val_transform
from encoding import ModalityEncoders
from fusionModel import FusionModel


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

CONFIG = {
    "checkpoint_path": "D:\\RWoodzell Classification Challenge\\checkpoints\\best_model.pth",
    "val_sar_root":    "D:\\RWoodzell Classification Challenge\\val",
    "val_csv_path":    "D:\\RWoodzell Classification Challenge\\val\\validation_reference.csv",
    "batch_size":      256,
    "num_workers":     16,
}


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────

class MAVICModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoders     = ModalityEncoders(freeze_eo_backbone=False)
        self.fusion_model = FusionModel(num_classes=num_classes)

    def forward(self, eo, sar):
        eo_feat, sar_feat = self.encoders(eo, sar)
        return self.fusion_model(eo_feat, sar_feat)


# ─────────────────────────────────────────────
#  OOD SCORING METHODS
# ─────────────────────────────────────────────

def score_energy(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """
    Energy score: E(x) = -T * logsumexp(logits / T)
    Higher energy = more likely OOD.
    Negate so that higher score = more OOD (consistent with other methods).
    """
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy.cpu().numpy()   # higher = more OOD


def score_softmax(logits: torch.Tensor) -> np.ndarray:
    """
    Softmax confidence: max(softmax(logits))
    Higher confidence = more likely in-distribution.
    Negate so that higher score = more OOD.
    """
    probs = F.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values
    return -confidence.cpu().numpy()   # negate: higher = more OOD


def score_mls(logits: torch.Tensor) -> np.ndarray:
    """
    Maximum Logit Score (MLS): max(logits) — no softmax applied.
    Higher max logit = more likely in-distribution.
    Negate so that higher score = more OOD.

    Why MLS?
        Softmax can suppress large logit differences due to normalization.
        MLS uses raw logits directly, which can better separate OOD samples
        whose logits are uniformly low vs in-dist samples with one high logit.
    """
    max_logit = logits.max(dim=1).values
    return -max_logit.cpu().numpy()   # negate: higher = more OOD


def score_entropy(logits: torch.Tensor) -> np.ndarray:
    """
    Entropy: H(x) = -sum(p * log(p))
    Higher entropy = more uncertain = more likely OOD.
    No negation needed — entropy is already higher for OOD.

    Why entropy?
        Measures how spread out the probability distribution is.
        In-dist: one class dominates → low entropy
        OOD:     probabilities spread across classes → high entropy
    """
    probs   = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return entropy.cpu().numpy()   # higher = more OOD


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def tnr_at_tpr95(ood_labels: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    TNR@TPR95 = True Negative Rate when True Positive Rate = 95%.
    This is exactly what the competition scores.
    TNR@TPR95 = 1 - FPR@95%TPR

    Higher is better. 1.0 = perfect, 0.0 = worst.

    Args:
        ood_labels : ground truth (1=OOD, 0=in-dist)
        ood_scores : higher score = more likely OOD
    """
    thresholds = np.sort(ood_scores)
    best_tnr   = 0.0

    for thresh in thresholds:
        predicted_ood = (ood_scores >= thresh).astype(int)

        tp = np.sum((predicted_ood == 1) & (ood_labels == 1))
        fn = np.sum((predicted_ood == 0) & (ood_labels == 1))
        fp = np.sum((predicted_ood == 1) & (ood_labels == 0))
        tn = np.sum((predicted_ood == 0) & (ood_labels == 0))

        tpr = tp / (tp + fn + 1e-8)
        tnr = tn / (tn + fp + 1e-8)   # TNR = 1 - FPR

        if tpr >= 0.95:
            best_tnr = tnr
            break

    return best_tnr


# ─────────────────────────────────────────────
#  MAIN COMPARISON
# ─────────────────────────────────────────────

def compare_ood_methods(config: dict):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Load Checkpoint ---
    checkpoint   = torch.load(config["checkpoint_path"], map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    num_classes  = len(class_to_idx)
    print(f"Loaded model — val acc: {checkpoint['val_acc']*100:.2f}%\n")

    # --- Load Model ---
    model = MAVICModel(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

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

    # --- Collect All Logits ---
    print("Running inference...")
    all_logits   = []
    all_ood_flags = []
    all_labels   = []
    all_preds    = []

    with torch.no_grad():
        for batch in val_loader:
            eo       = batch['eo'].to(device,    non_blocking=True)
            sar      = batch['sar'].to(device,    non_blocking=True)
            ood_flag = batch['ood_flag']
            labels   = batch['label']

            logits = model(eo, sar)
            preds  = logits.argmax(dim=1)

            all_logits.append(logits.cpu())
            all_ood_flags.extend(ood_flag.numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    all_logits    = torch.cat(all_logits, dim=0)   # [N, num_classes]
    all_ood_flags = np.array(all_ood_flags)         # [N] — 1=OOD, 0=in-dist
    all_labels    = np.array(all_labels)
    all_preds     = np.array(all_preds)

    # --- Classification Accuracy (in-dist only) ---
    in_dist_mask = (all_ood_flags == 0)
    acc = (all_preds[in_dist_mask] == all_labels[in_dist_mask]).mean()
    print(f"Classification Accuracy (in-dist): {acc*100:.2f}%\n")

    # --- Compute All OOD Scores ---
    scores = {
        "Energy":    score_energy(all_logits),
        "Softmax":   score_softmax(all_logits),
        "MLS":       score_mls(all_logits),
        "Entropy":   score_entropy(all_logits),
    }

    # --- Evaluate Each Method ---
    print("=" * 62)
    print(f"{'Method':<12} | {'AUROC':>8} | {'AUPR':>8} | {'TNR@TPR95':>10} | {'Est.Score':>10}")
    print("=" * 62)

    results = []
    for method_name, ood_scores in scores.items():
        auroc     = roc_auc_score(all_ood_flags, ood_scores)
        aupr      = average_precision_score(all_ood_flags, ood_scores)
        tnr95     = tnr_at_tpr95(all_ood_flags, ood_scores)

        # Estimated competition total score
        est_total = acc + auroc + tnr95

        print(
            f"{method_name:<12} | "
            f"{auroc*100:>7.2f}% | "
            f"{aupr*100:>7.2f}% | "
            f"{tnr95*100:>9.2f}% | "
            f"{est_total:>10.4f}"
        )

        results.append({
            "method":    method_name,
            "auroc":     auroc,
            "aupr":      aupr,
            "tnr@tpr95": tnr95,
            "est_total": est_total,
        })

    print("=" * 62)

    # --- Best Method ---
    best = max(results, key=lambda x: x["est_total"])
    print(f"\n🏆 Best method: {best['method']}")
    print(f"   AUROC      : {best['auroc']*100:.2f}%")
    print(f"   TNR@TPR95  : {best['tnr@tpr95']*100:.2f}%")
    print(f"   Est. Total : {best['est_total']:.4f}")
    print(f"\n→ Use '{best['method']}' as your score in results.csv")
    print(f"  Update submit_inference.py accordingly.")

    # --- Save comparison to CSV ---
    results_df = pd.DataFrame(results)
    out_path   = config["checkpoint_path"].replace("best_model.pth", "ood_comparison.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nFull comparison saved to: {out_path}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    compare_ood_methods(CONFIG)