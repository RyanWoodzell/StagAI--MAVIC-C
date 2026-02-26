"""
====================================================
MAVIC-C | Step 8: Training Loop
====================================================
Optimized for:
  - 2x NVIDIA RTX 6000 Ada (47.5 GB VRAM each)
  - AMD Ryzen Threadripper PRO 5995WX (64 cores / 128 threads)

Multi-GPU strategy: DataParallel
  Splits each batch across both GPUs automatically.
  With batch_size=128, each GPU processes 64 images at once.

CPU DataLoader workers: 16
  Uses 16 of your 128 logical processors for data loading.
  Keeps both GPUs fed without bottlenecking on I/O.
====================================================
"""

import os
import warnings
import logging

# Suppress non-critical torch warnings
warnings.filterwarnings("ignore", message=".*Graph break.*")
warnings.filterwarnings("ignore", message=".*Profiler function.*")
warnings.filterwarnings("ignore", message=".*NCCL.*")
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from preprocessing import build_dataloaders
from encoding import ModalityEncoders
from fusionModel import FusionModel


# ─────────────────────────────────────────────
#  FULL MODEL WRAPPER
# ─────────────────────────────────────────────

class MAVICModel(nn.Module):
    """
    Full end-to-end model:
        EO  [B, 3, 224, 224] ──┐
                                ├─ Encoders → Projection → Dropout → Fusion → Head
        SAR [B, 1, 224, 224] ──┘
    """

    def __init__(
        self,
        num_classes:        int   = 10,
        freeze_eo_backbone: bool  = False,   
        drop_prob:          float = 0.25,
        label_smooth:       float = 0.1,
        # SARCLIP INTEGRATION
        freeze_sar_backbone: bool = True,
        sar_weights_path:    str  = '',
    ):
        super().__init__()
        # BUG FIXED: ModalityEncoder → ModalityEncoders, and passes freeze_eo_backbone
        # SARCLIP INTEGRATION: pass SAR weights path and freeze flag
        self.encoders     = ModalityEncoders(
            freeze_eo_backbone  = freeze_eo_backbone,
            freeze_sar_backbone = freeze_sar_backbone,
            sar_weights_path    = sar_weights_path,
        )
        # SARCLIP INTEGRATION: sar_in_dim=768 (ViT-L-14) vs eo_in_dim=1280 (EfficientNet)
        self.fusion_model = FusionModel(
            num_classes  = num_classes,
            drop_prob    = drop_prob,
            label_smooth = label_smooth,
            eo_in_dim    = self.encoders.eo_feature_dim,
            sar_in_dim   = self.encoders.sar_feature_dim,
        )

    def forward(self, eo: torch.Tensor, sar: torch.Tensor):
        eo_feat, sar_feat = self.encoders(eo, sar)
        logits = self.fusion_model(eo_feat, sar_feat)
        return logits

    def compute_loss(self, logits, labels):
        return self.fusion_model.compute_loss(logits, labels)


# ─────────────────────────────────────────────
#  TRAINING CONFIG
# ─────────────────────────────────────────────

CONFIG = {
    # --- Paths ---
    "train_sar_root": "C:\\train\\SAR_Train",
    "train_eo_root":  "C:\\train\\EO_Train",
    "val_sar_root":   "D:\\RWoodzell Classification Challenge\\val",
    "val_csv_path":   "D:\\RWoodzell Classification Challenge\\val\\validation_reference.csv",
    "checkpoint_dir": "D:\\RWoodzell Classification Challenge\\#FinalTryModels",

    # --- Training (Optimized for 2x RTX 6000 Ada = 95GB VRAM) ---
    "epochs":        75,
    "batch_size":    512,           
    # Optimized for NVMe SSD; reduce to 4 if still on HDD
    "num_workers":   16,            # 64 cores - leave room for GPU threads
    "prefetch_factor": 4,           # Prefetch batches per worker
    
    "learning_rate": 3e-4,          # Higher LR with larger batch + OneCycleLR
    "weight_decay":  1e-4,
    "warmup_pct":    0.1,           # 10% warmup epochs

    # --- Model ---
    "drop_prob":          0.25,
    "label_smooth":       0.1,
    "freeze_eo_backbone": False,

    # SARCLIP INTEGRATION
    "freeze_sar_backbone": True,
    "sar_weights_path":    r"D:\RWoodzell Classification Challenge\BestModelTryAgain\StagAI--MAVIC-C\sar_clip\model_configs\ViT-L-14\models--BiliSakura--SARCLIP-ViT-L-14\snapshots\fd6c03457e79e65285acf0045f63ce6bc485650f\model.safetensors",
    "sar_in_dim":          768,
    "sar_unfreeze_epoch":   65,   

    # --- Hardware (RTX 6000 Ada Optimizations) ---
    "use_multi_gpu":   False,
    "use_compile":     False,       # Disabled: causes graph breaks with this model
    "use_tf32":        True,        # TF32 tensor cores on Ada
    "cudnn_benchmark": True,        # Optimize conv algorithms
}


# ─────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def validate(model, loader, device, use_multi_gpu):
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            eo       = batch['eo'].to(device,   non_blocking=True)
            sar      = batch['sar'].to(device,   non_blocking=True)
            labels   = batch['label'].to(device,  non_blocking=True)
            ood_flag = batch['ood_flag'].to(device, non_blocking=True)

            logits = model(eo, sar)

            # Only score in-distribution samples — OOD samples have label=-1
            in_dist_mask = (ood_flag == 0)
            if in_dist_mask.sum() == 0:
                continue

            in_dist_logits = logits[in_dist_mask]
            in_dist_labels = labels[in_dist_mask]

            if use_multi_gpu:
                loss = model.module.compute_loss(in_dist_logits, in_dist_labels)
            else:
                loss = model.compute_loss(in_dist_logits, in_dist_labels)

            preds          = in_dist_logits.argmax(dim=1)
            total_correct += (preds == in_dist_labels).sum().item()
            total_samples += in_dist_labels.size(0)
            total_loss    += loss.item() * in_dist_labels.size(0)

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


def train(config: dict):

    # --- Device Setup ---
    if not torch.cuda.is_available():
        print("⚠️  No CUDA GPUs found — training on CPU (will be slow)")
        device        = torch.device("cpu")
        use_multi_gpu = False
    else:
        num_gpus      = torch.cuda.device_count()
        device        = torch.device("cuda:0")
        use_multi_gpu = config["use_multi_gpu"] and num_gpus > 1
        
        # ═══════════════════════════════════════════════════════════════
        # RTX 6000 Ada Optimizations — TF32 + cuDNN benchmark
        # ═══════════════════════════════════════════════════════════════
        if config.get("use_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True   # ~2x faster matmul
            torch.backends.cudnn.allow_tf32       = True   # ~2x faster convs
            print("✅ TF32 enabled (Ada tensor cores)")
        
        if config.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True          # Auto-tune conv algorithms
            print("✅ cuDNN benchmark enabled")
        
        # Set memory allocation strategy for large batches
        torch.cuda.empty_cache()
        
        print(f"\nFound {num_gpus} GPU(s):")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
        if use_multi_gpu:
            print(f"  → Using DataParallel across all {num_gpus} GPUs\n")
        else:
            print(f"  → Using single GPU: {torch.cuda.get_device_name(0)}\n")

    # --- Data ---
    print("Loading datasets...")
    train_loader, val_loader, class_to_idx = build_dataloaders(
        train_sar_root = config["train_sar_root"],
        train_eo_root  = config["train_eo_root"],
        val_sar_root   = config["val_sar_root"],
        val_csv_path   = config["val_csv_path"],
        batch_size     = config["batch_size"],
        num_workers    = config["num_workers"],
    )

    num_classes = len(class_to_idx)
    print(f"Classes ({num_classes}): {class_to_idx}\n")

    # --- Model ---
    model = MAVICModel(
        num_classes        = num_classes,
        freeze_eo_backbone = config["freeze_eo_backbone"],  
        drop_prob          = config["drop_prob"],
        label_smooth       = config["label_smooth"],
        # SARCLIP INTEGRATION
        freeze_sar_backbone = config["freeze_sar_backbone"],
        sar_weights_path    = config["sar_weights_path"],
    )

    if use_multi_gpu:
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # ═══════════════════════════════════════════════════════════════
    # torch.compile — ~20-30% speedup on Ada architecture
    # ═══════════════════════════════════════════════════════════════
    if config.get("use_compile", False) and torch.cuda.is_available():
        try:
            # reduce-overhead mode is best for training loops
            model = torch.compile(model, mode="reduce-overhead")
            print("✅ torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"⚠️  torch.compile failed, continuing without: {e}")

    # Mixed precision — faster on Ada architecture, ~2x speedup
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    print(f"Batch size per GPU: {config['batch_size'] // max(1, torch.cuda.device_count())}\n")

    # --- Optimizer & Scheduler (OneCycleLR with warmup) ---
    optimizer = AdamW(
        model.parameters(),
        lr           = config["learning_rate"],
        weight_decay = config["weight_decay"],
        fused        = torch.cuda.is_available(),  # Fused AdamW for CUDA
    )
    
    # OneCycleLR: warmup → peak → anneal (better than plain CosineAnnealing)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr          = config["learning_rate"],
        epochs          = config["epochs"],
        steps_per_epoch = steps_per_epoch,
        pct_start       = config.get("warmup_pct", 0.1),  # 10% warmup
        anneal_strategy = 'cos',
        div_factor      = 25.0,      # initial_lr = max_lr / 25
        final_div_factor= 1e4,       # final_lr = max_lr / 10000
    )

    # --- Checkpointing ---
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_val_acc   = 0.0
    best_ckpt_path = os.path.join(config["checkpoint_dir"], "best_model.pth")

    # --- Training Loop ---
    print("=" * 100)
    print(f"{'Epoch':>6} | {'Loss(EO+SAR)':>12} | {'Loss(SAR)':>9} | {'Acc(EO+SAR)':>11} | {'Acc(SAR)':>8} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print("=" * 100)

    for epoch in range(1, config["epochs"] + 1):

        # ─────────────────────────────────────────────
        # SARCLIP BACKBONE FREEZE SCHEDULE
        # Frozen for first N epochs, then unfreeze for fine-tuning
        # ─────────────────────────────────────────────
        unfreeze_epoch = config.get("sar_unfreeze_epoch", 11)

        if epoch == unfreeze_epoch:
            m = model.module if use_multi_gpu else model
            for param in m.encoders.sar_encoder.backbone.parameters():
                param.requires_grad = True
            print(f"\n🔓 Epoch {epoch}: SARCLIP backbone unfrozen for fine-tuning")

            # Reset optimizer so newly unfrozen params get proper learning rate
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr           = config["learning_rate"] * 0.1,
                weight_decay = config["weight_decay"],
                fused        = torch.cuda.is_available(),
            )
            print(f"   Optimizer reset with fine-tune lr: {config['learning_rate'] * 0.1:.2e}")

        # --- Train ---
        model.train()
        total_loss = total_correct = total_samples = 0
        # EO-present tracking (eo_dropped == 0)
        eo_loss = eo_correct = eo_samples = 0
        # SAR-only tracking (eo_dropped == 1)
        sar_loss = sar_correct = sar_samples = 0

        for batch in train_loader:
            eo         = batch['eo'].to(device,         non_blocking=True)
            sar        = batch['sar'].to(device,         non_blocking=True)
            labels     = batch['label'].to(device,       non_blocking=True)
            eo_dropped = batch['eo_dropped'].to(device,  non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(eo, sar)
                if use_multi_gpu:
                    loss = model.module.compute_loss(logits, labels)
                else:
                    loss = model.compute_loss(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Step scheduler per batch (OneCycleLR requirement)
            scheduler.step()

            preds          = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss    += loss.item() * labels.size(0)

            # Split stats for EO-present vs SAR-only batches
            eo_mask  = (eo_dropped == 0)   # samples where EO was kept
            sar_mask = (eo_dropped == 1)   # samples where EO was zeroed
            if eo_mask.sum() > 0:
                eo_loss    += loss.item() * eo_mask.sum().item()
                eo_correct += (preds[eo_mask] == labels[eo_mask]).sum().item()
                eo_samples += eo_mask.sum().item()
            if sar_mask.sum() > 0:
                sar_loss    += loss.item() * sar_mask.sum().item()
                sar_correct += (preds[sar_mask] == labels[sar_mask]).sum().item()
                sar_samples += sar_mask.sum().item()

        train_loss = total_loss    / total_samples
        train_acc  = total_correct / total_samples
        eo_loss_avg  = eo_loss  / eo_samples  if eo_samples  > 0 else float('nan')
        sar_loss_avg = sar_loss / sar_samples if sar_samples > 0 else float('nan')
        eo_acc_avg   = eo_correct  / eo_samples  if eo_samples  > 0 else float('nan')
        sar_acc_avg  = sar_correct / sar_samples if sar_samples > 0 else float('nan')

        # --- Validate ---
        val_loss, val_acc = validate(model, val_loader, device, use_multi_gpu)

        # OneCycleLR steps per batch, not per epoch
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"{epoch:>6} | "
            f"{eo_loss_avg:>12.4f} | "
            f"{sar_loss_avg:>9.4f} | "
            f"{eo_acc_avg*100:>10.2f}% | "
            f"{sar_acc_avg*100:>7.2f}% | "
            f"{val_loss:>8.4f} | "
            f"{val_acc*100:>6.2f}% | "
            f"{current_lr:>8.2e}"
        )

        # Unwrap DataParallel before saving
        state = model.module if use_multi_gpu else model
        ckpt = {
            "epoch":        epoch,
            "model_state":  state.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "val_acc":      val_acc,
            "class_to_idx": class_to_idx,
        }

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_ckpt_path)
            print(f"         ✅ New best saved (val acc: {val_acc*100:.2f}%)")

        # Save every 5 epochs
        if epoch % 5 == 0:
            epoch_ckpt_path = os.path.join(config["checkpoint_dir"], f"model_epoch_{epoch}.pth")
            torch.save(ckpt, epoch_ckpt_path)
            print(f"         💾 Epoch {epoch} checkpoint saved")

    print("=" * 70)
    print(f"Training complete. Best val accuracy: {best_val_acc*100:.2f}%")
    print(f"Best model saved to: {best_ckpt_path}")
    return best_ckpt_path


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    best_model_path = train(CONFIG)
    print(f"\n✅ Step 8 complete. Ready for Step 7: OOD Detection.")
    print(f"   Load your trained model from: {best_model_path}")