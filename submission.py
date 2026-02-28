"""
====================================================
MAVIC-C | Submission: Generate results.csv
====================================================
Runs inference on the test folder (SAR-only images)
and generates a competition-ready results.csv.

Output format:
    image_id, class_id, score
    Gotcha12345, 2, 5.4
    ...

Score = Energy Score = -logsumexp(logits)
        Higher energy = more uncertain = more likely OOD.
        Lower energy  = more confident = more likely in-distribution.
====================================================
"""


import os
import time
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# SARCLIP INTEGRATION: use CLIP-normalized transform for inference
from preprocessing import sar_clip_val_transform as sar_val_transform
from encoding import ModalityEncoders
from fusionModel import FusionModel

# SARCLIP INTEGRATION
SAR_WEIGHTS_PATH = r"D:\RWoodzell Classification Challenge\BestModelTryAgain\StagAI--MAVIC-C\sar_clip\model_configs\ViT-L-14\models--BiliSakura--SARCLIP-ViT-L-14\snapshots\fd6c03457e79e65285acf0045f63ce6bc485650f\model.safetensors"


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

CONFIG = {
    "checkpoint_path": "D:\\RWoodzell Classification Challenge\\SARFOUNDATIONTryModels\\model_epoch_10.pth",
    "test_sar_root":   "D:\\RWoodzell Classification Challenge\\test",
    "output_dir":      "D:\\RWoodzell Classification Challenge\\Submissions\\SARFoundationSubmission1",
    "batch_size":      256,
    "num_workers":     16,
    "temperature":     1.0,   # energy temperature — 1.0 is standard default
}


# ─────────────────────────────────────────────
#  TEST DATASET
# ─────────────────────────────────────────────

class TestDataset(Dataset):
    """
    Loads test SAR images from a flat folder.
    Expects filenames like: Gotcha12345.png
    image_id extracted as:  Gotcha12345
    """

    def __init__(self, sar_root: str, transform=None):
        self.sar_root  = sar_root
        self.transform = transform

        valid_exts   = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        self.samples = []

        for fname in sorted(os.listdir(sar_root)):
            if not any(fname.lower().endswith(ext) for ext in valid_exts):
                continue
            stem     = os.path.splitext(fname)[0]
            image_id = ''.join(filter(str.isdigit, stem))
            image_path = os.path.join(sar_root, fname)
            self.samples.append((image_path, image_id))

        print(f"Test dataset: {len(self.samples)} images found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, image_id = self.samples[idx]

        sar_image = Image.open(image_path).convert('L')
        if self.transform:
            sar_image = self.transform(sar_image)

        eo_dummy = torch.zeros(3, 224, 224)

        return {
            'sar':      sar_image,
            'eo':       eo_dummy,
            'image_id': image_id,
        }


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────


# SARCLIP INTEGRATION: ViT-L-14 SAR encoder, fully unfrozen at inference
class MAVICModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoders = ModalityEncoders(
            freeze_eo_backbone  = False,
            freeze_sar_backbone = False,
            sar_weights_path    = SAR_WEIGHTS_PATH,
        )
        self.fusion_model = FusionModel(
            num_classes = num_classes,
            sar_in_dim  = 768,
        )

    def forward(self, eo, sar):
        eo_feat, sar_feat = self.encoders(eo, sar)
        return self.fusion_model(eo_feat, sar_feat)


# ─────────────────────────────────────────────
#  ENERGY SCORE
# ─────────────────────────────────────────────

def compute_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Energy Score: E(x) = -T * logsumexp(logits / T)

    Properties:
        Low energy  → confident prediction → in-distribution
        High energy → uncertain prediction → likely OOD

    The competition uses the score column for OOD ranking,
    so we pass the raw energy value directly.
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


# ─────────────────────────────────────────────
#  MAIN INFERENCE
# ─────────────────────────────────────────────

def run_inference(config: dict):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Load Checkpoint ---
    print("Loading checkpoint...")
    checkpoint   = torch.load(config["checkpoint_path"], map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    num_classes  = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"Classes ({num_classes}): {class_to_idx}")
    print(f"Best val acc: {checkpoint['val_acc']*100:.2f}%\n")

    # --- Load Model ---
    model = MAVICModel(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    # --- Test Dataset ---
    test_dataset = TestDataset(
        sar_root  = config["test_sar_root"],
        transform = sar_val_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = config["num_workers"],
        pin_memory  = True,
    )

    # --- Inference ---
    print("Running inference...")
    results      = []
    total_time   = 0.0
    total_images = 0

    with torch.no_grad():
        for batch in test_loader:
            eo        = batch['eo'].to(device,  non_blocking=True)
            sar       = batch['sar'].to(device,  non_blocking=True)
            image_ids = batch['image_id']

            t_start      = time.time()
            logits       = model(eo, sar)
            t_end        = time.time()

            total_time   += (t_end - t_start)
            total_images += sar.size(0)

            pred_class   = logits.argmax(dim=1)
            import torch.nn.functional as F
            probs        = F.softmax(logits, dim=1)
            softmax_score = probs.max(dim=1).values  

            for i in range(len(image_ids)):
                results.append({
                    "image_id": image_ids[i],
                    "class_id": pred_class[i].item(),
                    "score":    round(softmax_score[i].item(), 6),
                })

    runtime_per_image = total_time / total_images
    print(f"Processed {total_images} images")
    print(f"Runtime per image: {runtime_per_image:.4f}s")

    # --- Save results.csv ---
    os.makedirs(config["output_dir"], exist_ok=True)

    results_df = pd.DataFrame(results)
    csv_path   = os.path.join(config["output_dir"], "results.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\nresults.csv saved to: {csv_path}")
    print(f"Total rows          : {len(results_df)}")

    print(f"\nSample predictions:")
    print(results_df.head(10).to_string(index=False))

    print(f"\nPrediction distribution:")
    class_counts = results_df["class_id"].map(idx_to_class).value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls:<25}: {count:>5} ({count/len(results_df)*100:.1f}%)")

    print(f"\nEnergy score stats:")
    print(f"  Mean  : {results_df['score'].mean():.4f}")
    print(f"  Std   : {results_df['score'].std():.4f}")
    print(f"  Min   : {results_df['score'].min():.4f}")
    print(f"  Max   : {results_df['score'].max():.4f}")

    return runtime_per_image, csv_path


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    runtime_per_image, csv_path = run_inference(CONFIG)
    print(f"\n✅ Inference complete.")
    print(f"   Next: run build_submission.py to create submission.zip")
    print(f"   Runtime per image: {runtime_per_image:.4f}s  ← update build_submission.py")