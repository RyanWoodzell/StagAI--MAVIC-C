"""
====================================================
MAVIC-C | Step 1: Data Preprocessing
====================================================
EO  → Resize → RandomCrop → Flip → ColorJitter → ImageNet Normalize
SAR → Log-transform → Normalize → Light Augmentations

TWO dataset classes:
  EOSARDataset     → paired EO+SAR for training
  SAROnlyDataset   → SAR only + CSV ground truth for validation
====================================================
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────
#  SAR CUSTOM TRANSFORMS
# ─────────────────────────────────────────────

class SARLogTransform:
    """
    Log-compress SAR intensity values to tame the high dynamic range.
    Applied AFTER ToTensor(), so input is a float tensor.
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = torch.clamp(img, min=self.eps)
        img = torch.log1p(img)
        img = img / (img.max() + self.eps)
        return img


class SARNormalize:
    """
    Normalize SAR tensor with dataset-specific mean and std.
    ⚠️  Replace mean/std with values from compute_sar_stats()
        once you've run it on your training data.
    """
    def __init__(self, mean: float = 0.3, std: float = 0.15):
        self.mean = mean
        self.std  = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / (self.std + 1e-8)


# ─────────────────────────────────────────────
#  EO TRANSFORMS
# ─────────────────────────────────────────────

eo_train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eo_val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ─────────────────────────────────────────────
#  SAR TRANSFORMS
# ─────────────────────────────────────────────

sar_train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    SARLogTransform(eps=1e-6),
    SARNormalize(mean=0.3, std=0.15),
])

sar_val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    SARLogTransform(eps=1e-6),
    SARNormalize(mean=0.3, std=0.15),
])


# ─────────────────────────────────────────────
#  TRAINING DATASET — Paired EO + SAR
# ─────────────────────────────────────────────

# Official MAVIC-C competition class ordering.
# Hardcoded so training IDs always match submission IDs — no remapping needed.
COMPETITION_CLASS_TO_IDX = {
    "sedan":                  0,
    "SUV":                    1,
    "pickup_truck":           2,
    "van":                    3,
    "box_truck":              4,
    "motorcycle":             5,
    "flatbed_truck":          6,
    "bus":                    7,
    "pickup_truck_w_trailer": 8,
    "semi_w_trailer":         9,
}


def get_class_to_idx(root):
    """
    Returns the competition-fixed class→index mapping.
    Validates that every folder in `root` appears in the competition list
    so misnamed folders are caught immediately at startup.
    """
    found = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    unknown = [c for c in found if c not in COMPETITION_CLASS_TO_IDX]
    if unknown:
        raise ValueError(
            f"Folders not in competition class list: {unknown}\n"
            f"Expected: {list(COMPETITION_CLASS_TO_IDX.keys())}"
        )
    return COMPETITION_CLASS_TO_IDX


class EOSARDataset(Dataset):
    """
    Paired EO + SAR training dataset.

    Expects matching folder structure:
        <eo_root>/class_a/img_001.png ...
        <sar_root>/class_a/img_001.png ...  ← same filenames as EO

    Returns a dict with keys: 'eo', 'sar', 'label'
    """

    def __init__(self, sar_root, eo_root, sar_transform=None, eo_transform=None):
        self.sar_root      = sar_root
        self.eo_root       = eo_root
        self.sar_transform = sar_transform
        self.eo_transform  = eo_transform

        self.class_to_idx = get_class_to_idx(eo_root)
        self.samples = []

        for class_name, label in self.class_to_idx.items():
            sar_class_dir = os.path.join(sar_root, class_name)
            eo_class_dir  = os.path.join(eo_root,  class_name)

            valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

            for fname in os.listdir(eo_class_dir):
                if not any(fname.lower().endswith(ext) for ext in valid_exts):
                    continue

                eo_path  = os.path.join(eo_class_dir,  fname)
                sar_path = os.path.join(sar_class_dir, fname)
                if not os.path.isfile(sar_path):
                    raise FileNotFoundError(
                        f"SAR file not found for EO image: {eo_path}\n"
                        f"Expected SAR at: {sar_path}"
                        )
                self.samples.append((sar_path, eo_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sar_path, eo_path, label = self.samples[idx]

        sar_image = Image.open(sar_path).convert('L')
        eo_image  = Image.open(eo_path).convert('RGB')

        if self.sar_transform:
            sar_image = self.sar_transform(sar_image)
        if self.eo_transform:
            eo_image = self.eo_transform(eo_image)

        return {
            'sar':   sar_image,
            'eo':    eo_image,
            'label': label
        }


# ─────────────────────────────────────────────
#  VALIDATION DATASET — SAR Only + CSV
# ─────────────────────────────────────────────

class SAROnlyDataset(Dataset):
    """
    SAR-only validation dataset loaded from a CSV file.

    CSV format:
        image_id,   class,      OOD_flag
        Gotcha123,  box_truck,  0
        Gotcha456,  unknown,    1   ← OOD sample, label will be -1

    Args:
        sar_root      : folder containing all SAR .png files
        csv_path      : path to the ground truth CSV
        class_to_idx  : dict mapping class name → integer index
                        MUST use the same mapping as your training set
        sar_transform : SAR transform pipeline
    """

    def __init__(
        self,
        sar_root:     str,
        csv_path:     str,
        class_to_idx: dict,
        sar_transform=None,
    ):
        self.sar_root      = sar_root
        self.sar_transform = sar_transform
        self.class_to_idx  = class_to_idx

        df = pd.read_csv(csv_path)

        required_cols = {'image_id', 'class', 'OOD_flag'}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV missing columns. Expected: {required_cols}\n"
                f"Found: {set(df.columns)}"
            )

        self.samples = []
        skipped = 0

        for _, row in df.iterrows():
            image_id   = str(row['image_id'])
            class_name = str(row['class'])
            ood_flag   = int(row['OOD_flag'])
            sar_path   = os.path.join(sar_root, image_id + '.png')

            if not os.path.isfile(sar_path):
                skipped += 1
                continue

            label = -1 if ood_flag == 1 else class_to_idx.get(class_name, -1)

            self.samples.append((sar_path, label, ood_flag))

        if skipped > 0:
            print(f"⚠️  Skipped {skipped} rows — SAR image files not found.")

        n_ind = sum(1 for _, _, f in self.samples if f == 0)
        n_ood = sum(1 for _, _, f in self.samples if f == 1)
        print(f"Validation dataset: {len(self.samples)} samples "
              f"({n_ind} in-distribution, {n_ood} OOD)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sar_path, label, ood_flag = self.samples[idx]

        sar_image = Image.open(sar_path).convert('L')
        if self.sar_transform:
            sar_image = self.sar_transform(sar_image)

        # No EO available — zeros tensor stands in for missing modality
        eo_dummy = torch.zeros(3, 224, 224)

        return {
            'sar':      sar_image,
            'eo':       eo_dummy,
            'label':    label,
            'ood_flag': ood_flag,
        }


# ─────────────────────────────────────────────
#  DATALOADER FACTORY
# ─────────────────────────────────────────────

def build_dataloaders(
    train_sar_root: str,
    train_eo_root:  str,
    val_sar_root:   str,
    val_csv_path:   str,          # ← accepts val_csv_path
    batch_size:     int = 32,
    num_workers:    int = 4,
):
    """
    Build train and val DataLoaders.

    Returns:
        train_loader, val_loader, class_to_idx
    """
    train_dataset = EOSARDataset(
        sar_root      = train_sar_root,
        eo_root       = train_eo_root,
        sar_transform = sar_train_transform,
        eo_transform  = eo_train_transform,
    )

    # Use the SAME class mapping from training for validation
    val_dataset = SAROnlyDataset(
        sar_root      = val_sar_root,
        csv_path      = val_csv_path,
        class_to_idx  = train_dataset.class_to_idx,   # must match training
        sar_transform = sar_val_transform,
    )

    # ═══════════════════════════════════════════════════════════════
    # DataLoader Optimizations for Threadripper PRO + NVMe
    # ═══════════════════════════════════════════════════════════════
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,              # Faster GPU transfer
        drop_last   = True,              # Consistent batch sizes
        persistent_workers = True,       # Keep workers alive between epochs
        prefetch_factor    = 4,          # Prefetch 4 batches per worker
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size * 2,    # Larger val batch (no gradients)
        shuffle     = False,
        num_workers = num_workers // 2,  # Fewer workers for val
        pin_memory  = True,
        persistent_workers = True,
        prefetch_factor    = 2,
    )

    return train_loader, val_loader, train_dataset.class_to_idx


# ─────────────────────────────────────────────
#  UTILITY: Compute SAR Dataset Statistics
# ─────────────────────────────────────────────

def compute_sar_stats(sar_root: str) -> dict:
    """
    Compute mean and std of SAR images AFTER log-transform.
    Run ONCE on your training SAR folder, then update SARNormalize above.

    Usage:
        stats = compute_sar_stats('D:/.../.../SAR_Train')
    """
    to_tensor     = transforms.ToTensor()
    log_transform = SARLogTransform()
    all_values    = []

    for class_name in os.listdir(sar_root):
        class_dir = os.path.join(sar_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            path   = os.path.join(class_dir, fname)
            img    = Image.open(path).convert('L')
            tensor = to_tensor(img)
            tensor = log_transform(tensor)
            all_values.append(tensor.flatten())

    all_values = torch.cat(all_values)
    stats = {
        "mean": round(all_values.mean().item(), 4),
        "std":  round(all_values.std().item(),  4),
    }
    print(f"SAR stats → mean: {stats['mean']}, std: {stats['std']}")
    print("Update SARNormalize(mean=..., std=...) with these values.")
    return stats


# ─────────────────────────────────────────────
#  EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    train_loader, val_loader, class_to_idx = build_dataloaders(
        train_sar_root = 'D:\\RWoodzell Classification Challenge\\train\\SAR_Train',
        train_eo_root  = 'D:\\RWoodzell Classification Challenge\\train\\EO_Train',
        val_sar_root   = 'D:\\RWoodzell Classification Challenge\\val\\SAR_Val',
        val_csv_path   = 'D:\\RWoodzell Classification Challenge\\val\\ground_truth.csv',
        batch_size     = 32,
    )

    print(f"\nClass mapping : {class_to_idx}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    batch = next(iter(val_loader))
    print(f"\nVal SAR shape  : {batch['sar'].shape}")
    print(f"Val EO shape   : {batch['eo'].shape}")
    print(f"Val labels     : {batch['label']}")
    print(f"Val OOD flags  : {batch['ood_flag']}")

    print("\n✅ Step 1 complete.")