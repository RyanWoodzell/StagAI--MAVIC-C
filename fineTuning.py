"""
Fine-tune best_model.pth with corrected class mapping.
Remaps the classification head to competition class order.
"""
import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
from preprocessing import build_dataloaders, COMPETITION_CLASS_TO_IDX
from encoding import ModalityEncoders
from fusionModel import FusionModel

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "checkpoint_path": r"D:\RWoodzell Classification Challenge\BEST MODEL\best_model.pth",
    "train_sar_root":  r"C:\train\SAR_Train",
    "train_eo_root":   r"C:\train\EO_Train",
    "val_sar_root":    r"D:\RWoodzell Classification Challenge\val",
    "val_csv_path":    r"D:\RWoodzell Classification Challenge\val\validation_reference.csv",
    "save_dir":        r"D:\RWoodzell Classification Challenge\checkpointsFinetuned",
    "epochs":          30,
    "batch_size":      512,
    "num_workers":     16,
    "learning_rate":   1e-4,   # Lower LR for fine-tuning
    "weight_decay":    1e-4,
    "warmup_pct":      0.1,
    "unfreeze_layers": "head_only",  # "head_only" | "last_block" | "full"
}

# ─────────────────────────────────────────────
#  OLD → NEW CLASS REMAP
# ─────────────────────────────────────────────
# Checkpoint used alphabetical order — competition uses frequency order
OLD_CLASS_TO_IDX = {
    'SUV': 0, 'box_truck': 1, 'bus': 2, 'flatbed_truck': 3,
    'motorcycle': 4, 'pickup_truck': 5, 'pickup_truck_w_trailer': 6,
    'sedan': 7, 'semi_w_trailer': 8, 'van': 9
}
NEW_CLASS_TO_IDX = COMPETITION_CLASS_TO_IDX  # competition order

# Maps old index → new index
OLD_TO_NEW = {
    OLD_CLASS_TO_IDX[cls]: NEW_CLASS_TO_IDX[cls]
    for cls in OLD_CLASS_TO_IDX
}
print("Class remap (old → new):", OLD_TO_NEW)


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


def remap_head_weights(state_dict: dict, old_to_new: dict) -> dict:
    """
    Reorder the classification head output weights to match new class order.
    Affects the final Linear layer: weight [num_classes, in_dim] and bias [num_classes].
    """
    # Find head weight/bias keys
    head_weight_key = None
    head_bias_key   = None
    for k in state_dict:
        if "classifier" in k and "weight" in k and state_dict[k].dim() == 2:
            if state_dict[k].shape[0] == len(old_to_new):
                head_weight_key = k
        if "classifier" in k and "bias" in k and state_dict[k].dim() == 1:
            if state_dict[k].shape[0] == len(old_to_new):
                head_bias_key = k

    if head_weight_key is None:
        print("⚠️  Could not find head weight key — skipping remap")
        return state_dict

    print(f"Remapping: {head_weight_key}, {head_bias_key}")

    old_weight = state_dict[head_weight_key].clone()
    old_bias   = state_dict[head_bias_key].clone()

    new_weight = torch.zeros_like(old_weight)
    new_bias   = torch.zeros_like(old_bias)

    for old_idx, new_idx in old_to_new.items():
        new_weight[new_idx] = old_weight[old_idx]
        new_bias[new_idx]   = old_bias[old_idx]

    state_dict[head_weight_key] = new_weight
    state_dict[head_bias_key]   = new_bias
    return state_dict


def set_trainable(model: nn.Module, mode: str):
    """
    Control which layers are trainable.
    head_only   → freeze all encoders, train fusion + head only
    last_block  → freeze early layers, train last encoder block + head
    full        → train everything (lowest LR)
    """
    if mode == "head_only":
        for param in model.encoders.parameters():
            param.requires_grad = False
        for param in model.fusion_model.parameters():
            param.requires_grad = True
        print("🔒 Encoders frozen — training fusion + head only")

    elif mode == "last_block":
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last encoder block + BN layers
        for param in model.encoders.eo_encoder.features[-1].parameters():
            param.requires_grad = True
        for param in model.encoders.sar_encoder.features[-1].parameters():
            param.requires_grad = True
        for param in model.fusion_model.parameters():
            param.requires_grad = True
        print("🔒 Early layers frozen — training last encoder block + head")

    elif mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        print("🔓 All layers unfrozen — full fine-tune")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")


# ─────────────────────────────────────────────
#  FINE-TUNE LOOP
# ─────────────────────────────────────────────
def finetune(config: dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(config["save_dir"], exist_ok=True)

    # --- Load checkpoint ---
    print(f"\nLoading checkpoint: {config['checkpoint_path']}")
    ckpt = torch.load(config["checkpoint_path"], map_location=device)
    state_dict = ckpt["model_state"]

    # --- Remap class head ---
    state_dict = remap_head_weights(state_dict, OLD_TO_NEW)

    # --- Build model ---
    model = MAVICModel(num_classes=10).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️  Missing keys: {missing}")
    if unexpected:
        print(f"⚠️  Unexpected keys: {unexpected}")
    print("✅ Checkpoint loaded and class head remapped")

    # --- Set trainable layers ---
    set_trainable(model, config["unfreeze_layers"])

    # --- Data ---
    train_loader, val_loader, class_to_idx = build_dataloaders(
        train_sar_root = config["train_sar_root"],
        train_eo_root  = config["train_eo_root"],
        val_sar_root   = config["val_sar_root"],
        val_csv_path   = config["val_csv_path"],
        batch_size     = config["batch_size"],
        num_workers    = config["num_workers"],
    )
    print(f"Classes: {class_to_idx}")

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = config["learning_rate"],
        weight_decay = config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr        = config["learning_rate"],
        steps_per_epoch = len(train_loader),
        epochs        = config["epochs"],
        pct_start     = config["warmup_pct"],
    )
    scaler = torch.cuda.amp.GradScaler()

    best_acc  = ckpt.get("val_acc", 0.0)
    best_path = os.path.join(config["save_dir"], "finetuned_best.pth")

    print(f"\nStarting fine-tune from epoch {ckpt['epoch']+1} | Baseline val acc: {best_acc*100:.2f}%")
    print("=" * 70)
    print(f" {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Acc':>7} | {'LR':>10}")
    print("=" * 70)

    for epoch in range(config["epochs"]):
        # ── Train ──
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            eo  = batch["eo"].to(device,  non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(eo, sar)
                loss   = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_loss += loss.item()
            correct  += (logits.argmax(1) == lbl).sum().item()
            total    += lbl.size(0)

        train_loss = run_loss / len(train_loader)
        train_acc  = correct / total

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                eo  = batch["eo"].to(device,  non_blocking=True)
                sar = batch["sar"].to(device, non_blocking=True)
                lbl = batch["label"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(eo, sar)
                val_correct += (logits.argmax(1) == lbl).sum().item()
                val_total   += lbl.size(0)

        val_acc = val_correct / val_total
        lr_now  = scheduler.get_last_lr()[0]

        print(f" {epoch+1:>5} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | {val_acc*100:>6.2f}% | {lr_now:>10.2e}")

        # ── Save best ──
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch":        epoch + ckpt["epoch"] + 1,
                "model_state":  model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "val_acc":      val_acc,
                "class_to_idx": NEW_CLASS_TO_IDX,
            }, best_path)
            print(f"   💾 Saved best model — val acc: {val_acc*100:.2f}%")

    print("=" * 70)
    print(f"✅ Fine-tuning complete. Best val acc: {best_acc*100:.2f}%")
    print(f"   Saved to: {best_path}")
    return best_path


if __name__ == "__main__":
    finetune(CONFIG)