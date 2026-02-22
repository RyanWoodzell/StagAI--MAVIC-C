"""
====================================================
MAVIC-C | Inference on Validation Dataset
====================================================
Tests trained .pth models on validation SAR images.

Supports:
  - In-distribution accuracy
  - OOD detection metrics (AUROC, F1-score)
  - Per-class performance
  - Confusion matrix

Usage:
    python inference_validation.py --model "path/to/model.pth" \
                                   --csv "path/to/validation_reference.csv" \
                                   --sar_root "path/to/sar/images"
====================================================
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)

# Import model components
from train import MAVICModel
from preprocessing import sar_val_transform, get_class_to_idx


# ─────────────────────────────────────────────
#  VALIDATION DATASET
# ─────────────────────────────────────────────

class ValidationDataset(Dataset):
    """
    Load SAR-only validation images from CSV.
    
    CSV format:
        image_id,   class,      OOD_flag
        Gotcha123,  box_truck,  0
        Gotcha456,  unknown,    1   ← OOD sample
    """
    
    def __init__(self, sar_root: str, csv_path: str, class_to_idx: dict,
                 sar_transform=None):
        self.sar_root = sar_root
        self.sar_transform = sar_transform
        self.class_to_idx = class_to_idx
        
        df = pd.read_csv(csv_path)
        self.samples = []
        self.skipped = []
        
        for _, row in df.iterrows():
            image_id = str(row['image_id']).strip()
            class_name = str(row['class']).strip()
            ood_flag = int(row['OOD_flag'])
            
            # Find image file (handle .png, .jpg, etc)
            sar_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                candidate = os.path.join(sar_root, image_id + ext)
                if os.path.isfile(candidate):
                    sar_path = candidate
                    break
            
            if sar_path is None:
                self.skipped.append(image_id)
                continue
            
            label = -1 if ood_flag == 1 else class_to_idx.get(class_name, -1)
            self.samples.append((sar_path, label, ood_flag, image_id))
        
        if self.skipped:
            print(f"⚠️  Skipped {len(self.skipped)} samples — image files not found")
            print(f"   First 5: {self.skipped[:5]}")
        
        n_ind = sum(1 for _, _, f, _ in self.samples if f == 0)
        n_ood = sum(1 for _, _, f, _ in self.samples if f == 1)
        print(f"✅ Validation dataset: {len(self.samples)} samples "
              f"({n_ind} in-distribution, {n_ood} OOD)\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sar_path, label, ood_flag, image_id = self.samples[idx]
        
        sar_image = Image.open(sar_path).convert('L')
        if self.sar_transform:
            sar_image = self.sar_transform(sar_image)
        
        # Dummy EO for model forward pass
        eo_dummy = torch.zeros(3, 224, 224)
        
        return {
            'sar': sar_image,
            'eo': eo_dummy,
            'label': label,
            'ood_flag': ood_flag,
            'image_id': image_id,
        }


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────

def load_model(model_path: str, num_classes: int, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = MAVICModel(num_classes=num_classes)
    
    # Handle DataParallel wrapper
    state_dict = checkpoint.get('model_state', checkpoint)
    if isinstance(state_dict, dict) and 'module.' in list(state_dict.keys())[0]:
        # Remove DataParallel wrapper prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def infer(model: nn.Module, loader: DataLoader, device: torch.device,
          class_to_idx: dict) -> dict:
    """Run inference on validation set."""
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    all_preds = []
    all_labels = []
    all_ood_flags = []
    all_logits = []
    all_image_ids = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            eo = batch['eo'].to(device, non_blocking=True)
            sar = batch['sar'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            ood_flags = batch['ood_flag'].to(device, non_blocking=True)
            image_ids = batch['image_id']
            
            # Forward pass
            logits = model(eo, sar)
            
            all_logits.append(logits.cpu())
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
            all_ood_flags.append(ood_flags.cpu())
            all_image_ids.extend(image_ids)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches")
    
    results = {
        'logits': torch.cat(all_logits, dim=0),
        'preds': torch.cat(all_preds, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'ood_flags': torch.cat(all_ood_flags, dim=0),
        'image_ids': all_image_ids,
    }
    
    return results


def compute_metrics(results: dict, class_to_idx: dict, idx_to_class: dict) -> dict:
    """Compute evaluation metrics."""
    
    preds = results['preds'].numpy()
    labels = results['labels'].numpy()
    ood_flags = results['ood_flags'].numpy()
    logits = results['logits'].numpy()
    
    # Filter: in-distribution only
    ind_mask = (ood_flags == 0) & (labels >= 0)
    
    preds_ind = preds[ind_mask]
    labels_ind = labels[ind_mask]
    logits_ind = logits[ind_mask]
    
    # In-distribution accuracy
    acc = accuracy_score(labels_ind, preds_ind)
    prec = precision_score(labels_ind, preds_ind, average='weighted', zero_division=0)
    recall = recall_score(labels_ind, preds_ind, average='weighted', zero_division=0)
    f1 = f1_score(labels_ind, preds_ind, average='weighted', zero_division=0)
    
    # OOD detection: max softmax confidence as OOD score
    max_probs = torch.softmax(torch.from_numpy(logits), dim=1).max(dim=1).values.numpy()
    ood_labels_binary = (ood_flags == 1).astype(int)
    
    try:
        auroc = roc_auc_score(ood_labels_binary, 1.0 - max_probs)
    except:
        auroc = 0.0
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': recall,
        'f1': f1,
        'n_ind': len(preds_ind),
        'n_ood': (ood_flags == 1).sum(),
        'auroc': auroc,
    }
    
    return metrics, (preds_ind, labels_ind, idx_to_class, preds, labels, ood_flags, logits)


def print_results(metrics: dict, class_results: dict = None):
    """Print evaluation results."""
    
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 In-Distribution Performance ({metrics['n_ind']} samples):")
    print(f"  Accuracy:  {metrics['accuracy']*100:>6.2f}%")
    print(f"  Precision: {metrics['precision']*100:>6.2f}%")
    print(f"  Recall:    {metrics['recall']*100:>6.2f}%")
    print(f"  F1-Score:  {metrics['f1']*100:>6.2f}%")
    
    print(f"\n🚨 OOD Detection ({metrics['n_ood']} OOD samples):")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    
    if class_results:
        print(f"\n🏆 Per-Class Accuracy:")
        for cls_name, acc in sorted(class_results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls_name:>25s}: {acc*100:>6.2f}%")
    
    print("\n" + "=" * 70)


def plot_confusion_matrix(preds_ind, labels_ind, idx_to_class, save_path: str = None):
    """Plot and save confusion matrix."""
    
    cm = confusion_matrix(labels_ind, preds_ind)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - In-Distribution Samples')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Test MAVIC-C model on validation dataset'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to .pth model file')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to validation_reference.csv')
    parser.add_argument('--sar_root', type=str, required=True,
                        help='Path to SAR validation images folder')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
    parser.add_argument('--save_cm', action='store_true',
                        help='Save confusion matrix plot')
    parser.add_argument('--results_csv', type=str, default=None,
                        help='Save detailed results to CSV')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not os.path.isdir(args.sar_root):
        raise FileNotFoundError(f"SAR folder not found: {args.sar_root}")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load checkpoint to get class info
    checkpoint = torch.load(args.model, map_location=device)
    class_to_idx = checkpoint.get('class_to_idx', None)
    
    if class_to_idx is None:
        # Infer from CSV
        df = pd.read_csv(args.csv)
        classes = sorted(set(df['class']) - {'unknown'})
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        print(f"⚠️  Inferred class mapping from CSV")
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    print(f"Model: {os.path.basename(args.model)}")
    print(f"Classes: {num_classes}")
    print(f"Class mapping: {class_to_idx}\n")
    
    # Build dataset
    val_dataset = ValidationDataset(
        sar_root=args.sar_root,
        csv_path=args.csv,
        class_to_idx=class_to_idx,
        sar_transform=sar_val_transform,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Load model
    print(f"Loading model...")
    model = load_model(args.model, num_classes, device)
    
    # Run inference
    results = infer(model, val_loader, device, class_to_idx)
    
    # Compute metrics
    metrics, extra_data = compute_metrics(results, class_to_idx, idx_to_class)
    
    # Per-class metrics
    preds_ind, labels_ind, _, _, _, _, _ = extra_data
    class_results = {}
    for cls_idx in sorted(idx_to_class.keys()):
        mask = labels_ind == cls_idx
        if mask.sum() > 0:
            class_results[idx_to_class[cls_idx]] = accuracy_score(
                labels_ind[mask], preds_ind[mask]
            )
    
    # Print results
    print_results(metrics, class_results)
    
    # Save confusion matrix
    if args.save_cm:
        cm_path = args.model.replace('.pth', '_cm.png')
        plot_confusion_matrix(preds_ind, labels_ind, idx_to_class, cm_path)
    
    # Save detailed results CSV
    if args.results_csv:
        results_df = pd.DataFrame({
            'image_id': results['image_ids'],
            'true_label': results['labels'].numpy(),
            'pred_label': results['preds'].numpy(),
            'ood_flag': results['ood_flags'].numpy(),
            'confidence': torch.softmax(results['logits'], dim=1).max(dim=1).values.numpy(),
        })
        results_df.to_csv(args.results_csv, index=False)
        print(f"✅ Detailed results saved to: {args.results_csv}")


if __name__ == "__main__":
    main()
