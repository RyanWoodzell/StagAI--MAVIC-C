"""
Neural Architecture Search via Optuna.
Searches over backbone size, projection dims, fusion type, dropout.
Each trial trains for `trial_epochs` and reports val accuracy.
"""
import os
import sys
import torch
import torch.nn as nn
import optuna
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

sys.path.append(os.path.dirname(__file__))
from preprocessing import build_dataloaders
from encoding import ModalityEncoders
from fusionModel import FusionModel

# Global — set before workers spawn so they inherit the correct GPU
GPU_ID = 1 if torch.cuda.device_count() > 1 else 0

def _worker_init(worker_id):
    """Pin each DataLoader worker process to the correct GPU."""
    torch.cuda.set_device(GPU_ID)

# ─────────────────────────────────────────────
#  NAS CONFIG
# ─────────────────────────────────────────────
NAS_CONFIG = {
    "train_sar_root": r"C:\train\SAR_Train",
    "train_eo_root":  r"C:\train\EO_Train",
    "val_sar_root":   r"D:\RWoodzell Classification Challenge\val",
    "val_csv_path":   r"D:\RWoodzell Classification Challenge\val\validation_reference.csv",
    "save_dir":       r"D:\RWoodzell Classification Challenge\NAS",
    "n_trials":       15,   # 15 trials ≈ 2.5 hrs total
    "trial_epochs":   3,    # 3 epochs is enough to rank architectures
    "batch_size":     256,  # B3+B2 need headroom; 256 still fast enough
    "num_workers":    4,
    "num_classes":    10,
}


# ─────────────────────────────────────────────
#  SEARCH SPACE
# ─────────────────────────────────────────────
def build_trial_model(trial: optuna.Trial, num_classes: int) -> nn.Module:
    """
    Optuna suggests values for each hyperparameter.
    Each call builds a different architecture variant.
    """

    # --- Backbone size ---
    backbone = trial.suggest_categorical(
        "backbone", ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"]
    )

    # --- Projection MLP ---
    proj_dim     = trial.suggest_categorical("proj_dim", [128, 256, 512])

    # --- Modality dropout ---
    drop_prob   = trial.suggest_float("drop_prob",   0.2, 0.6, step=0.1)

    # --- Head ---
    head_dropout = trial.suggest_float("head_dropout", 0.3, 0.5, step=0.1)
    label_smooth = trial.suggest_float("label_smooth", 0.05, 0.2, step=0.05)

    # --- Optimizer ---
    lr           = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    class TrialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders     = ModalityEncoders(
                freeze_eo_backbone = False,
                backbone           = backbone,      # ← now works
            )
            in_dim = self.encoders.feature_dim      # 1280/1280/1408/1536 depending on backbone
            self.fusion_model = FusionModel(
                num_classes  = num_classes,
                proj_dim     = proj_dim,
                drop_prob    = drop_prob,
                head_dropout = head_dropout,
                label_smooth = label_smooth,
                eo_in_dim    = in_dim,
                sar_in_dim   = in_dim,
            )

        def forward(self, eo, sar):
            eo_feat, sar_feat = self.encoders(eo, sar)
            return self.fusion_model(eo_feat, sar_feat)

        def compute_loss(self, logits, labels):
            return self.fusion_model.compute_loss(logits, labels)

    return TrialModel(), lr, weight_decay


# ─────────────────────────────────────────────
#  SINGLE TRIAL TRAINING
# ─────────────────────────────────────────────
def run_trial(trial: optuna.Trial, config: dict, device: torch.device,
              train_loader, val_loader) -> float:
    """Train one architecture for trial_epochs, return val accuracy."""

    model, lr, weight_decay = build_trial_model(trial, config["num_classes"])
    torch.cuda.empty_cache()  # clear fragmented memory before loading new model
    try:
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        print(f"  💥 OOM moving model to GPU — pruning trial", flush=True)
        del model; torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()

    print(f"\n▶ Trial {trial.number+1}/{config['n_trials']} | "
          f"backbone={trial.params['backbone']} | "
          f"proj_dim={trial.params['proj_dim']} | "
          f"drop={trial.params['drop_prob']:.2f} | "
          f"lr={lr:.2e}", flush=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr          = lr,
        epochs          = config["trial_epochs"],
        steps_per_epoch = len(train_loader),
        pct_start       = 0.1,
    )
    scaler = torch.amp.GradScaler(device.type)

    best_val = 0.0
    try:
        # ── Train ──
        for epoch in range(config["trial_epochs"]):
            model.train()
            run_loss = correct = total = 0
            for batch in train_loader:
                eo     = batch["eo"].to(device,    non_blocking=True)
                sar    = batch["sar"].to(device,   non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast(device.type):
                    logits = model(eo, sar)
                    loss   = model.compute_loss(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                run_loss += loss.item()
                correct  += (logits.argmax(1) == labels).sum().item()
                total    += labels.size(0)

            val_acc  = quick_val(model, val_loader, device)
            best_val = max(best_val, val_acc)

            print(f"  Epoch {epoch+1}/{config['trial_epochs']} | "
                  f"loss={run_loss/len(train_loader):.4f} | "
                  f"train={correct/total*100:.1f}% | "
                  f"val={val_acc*100:.1f}%", flush=True)

            # ── Prune bad trials early ──
            trial.report(val_acc, epoch)
            if trial.should_prune():
                print(f"  ✂️  Pruned at epoch {epoch+1}", flush=True)
                del model
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

    except torch.cuda.OutOfMemoryError:
        print(f"  💥 OOM during training — pruning trial (backbone too large for batch size)", flush=True)
        del model
        torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()

    del model
    torch.cuda.empty_cache()
    return best_val


def quick_val(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            eo     = batch["eo"].to(device,    non_blocking=True)
            sar    = batch["sar"].to(device,   non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            ood    = batch["ood_flag"].to(device, non_blocking=True)

            logits = model(eo, sar)
            mask   = (ood == 0)
            if mask.sum() == 0:
                continue
            correct += (logits[mask].argmax(1) == labels[mask]).sum().item()
            total   += mask.sum().item()

    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────
#  RUN SEARCH
# ─────────────────────────────────────────────
def run_nas(config: dict):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device(f"cuda:{GPU_ID}")
    torch.cuda.set_device(device)   # pin main process to correct GPU
    print(f"NAS running on: {device}")
    os.makedirs(config["save_dir"], exist_ok=True)

    # Pre-download all backbone weights so no trial blocks on a network request
    print("Pre-caching backbone weights...")
    from encoding import BACKBONE_REGISTRY
    for name, (model_fn, weights, _) in BACKBONE_REGISTRY.items():
        print(f"  {name}...", end=" ", flush=True)
        model_fn(weights=weights)   # triggers download if not already cached
        print("✅")
    print()

    # Build DataLoaders ONCE — reused across all 30 trials
    # Avoids re-scanning 455k files and re-spawning workers every trial
    print("Building DataLoaders (once)...")
    train_loader, val_loader, _ = build_dataloaders(
        train_sar_root = config["train_sar_root"],
        train_eo_root  = config["train_eo_root"],
        val_sar_root   = config["val_sar_root"],
        val_csv_path   = config["val_csv_path"],
        batch_size     = config["batch_size"],
        num_workers    = config["num_workers"],
        worker_init_fn = _worker_init,
    )
    print("DataLoaders ready.\n")

    # MedianPruner stops bad trials after a few epochs
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    db_path = os.path.join(config["save_dir"], "nas_results.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print("🗑️  Deleted old nas_results.db (search space changed)")

    study   = optuna.create_study(
        direction    = "maximize",
        pruner       = pruner,
        study_name   = "mavic_nas",
        storage      = f"sqlite:///{db_path}",
        load_if_exists = False,
    )

    def objective(trial):
        return run_trial(trial, config, device, train_loader, val_loader)

    study.optimize(objective, n_trials=config["n_trials"], show_progress_bar=True)

    # ── Results ──
    print("\n" + "=" * 60)
    print("NAS COMPLETE")
    print("=" * 60)
    print(f"Best val acc : {study.best_value*100:.2f}%")
    print(f"Best params  :")
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v}")

    # Save best config
    import json
    best_path = os.path.join(config["save_dir"], "best_config.json")
    with open(best_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest config saved to: {best_path}")
    return study.best_params


if __name__ == "__main__":
    best_params = run_nas(NAS_CONFIG)