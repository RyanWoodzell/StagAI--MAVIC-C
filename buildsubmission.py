"""
====================================================
MAVIC-C | Build Submission ZIP
====================================================
Packages results.csv and readme.txt into submission.zip
ready for competition upload.

Run AFTER submit_inference.py has generated results.csv.
====================================================
"""

import os
import zipfile

# ─────────────────────────────────────────────
#  CONFIG
#  Update runtime_per_image with value printed
#  at the end of submit_inference.py
# ─────────────────────────────────────────────

CONFIG = {
    "output_dir":      "D:\\RWoodzell Classification Challenge\\Submissions\\SARFoundationFinalAttempt",
    "runtime_per_image": 0.00011,   # ← from submit_inference.py output ############ THIS NEEDS TO BE UPDATED BEFORE RUNNING
    "uses_gpu":          0,         # 0 = GPU, 1 = CPU — you use GPU
    "extra_data":        0,         # 0 = no extra data used
    "description": (
        "Late fusion multimodal classification model for SAR-only inference. "
    "Architecture: EO encoder — EfficientNet-B0 pretrained on ImageNet (3-channel, 1280-dim output). "
    "SAR encoder — SARCLIP ViT-L-14 pretrained on SAR imagery via Contrastive Language-Image "
    "Pre-training (CLIP) framework, loaded from HuggingFace (BiliSakura/SARCLIP-ViT-L-14), "
    "single-channel input expanded to 3-channel by repetition, 768-dim output. "
    "EO features projected through 2-layer MLP (1280→768→256) with BatchNorm and GELU. "
    "SAR features projected through 2-layer MLP (768→512→256) with BatchNorm and GELU. "
    "EO and SAR projections concatenated to 512-dim fused representation. "
    "Modality dropout (p=0.25) applied during training — randomly zeros one modality per batch "
    "to force SAR-only robustness at test time. "
    "Classification head: Linear(512→256)→ReLU→Dropout→Linear(256→10). "
    "Training: SARCLIP backbone frozen for epochs 1-10, unfrozen at epoch 11 with lr=3e-5. "
    "AdamW optimizer with OneCycleLR scheduler, batch size 512, "
    "Focal Loss (gamma=2.0) with label smoothing (0.1), weighted random sampler for class balance. "
    "Mixed precision FP16, TF32 tensor cores enabled. "
    "Hardware: NVIDIA RTX 6000 Ada (51.5 GB VRAM), "
    "AMD Ryzen Threadripper PRO 5995WX 64-core. "
    ),
}
'''OOD detection via temperature scaled negative energy score
'''

def build_submission(config: dict):

    output_dir  = config["output_dir"]
    csv_path    = os.path.join(output_dir, "results.csv")
    readme_path = os.path.join(output_dir, "readme.txt")
    zip_path    = os.path.join(output_dir, "submission.zip")

    # --- Validate results.csv exists ---
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"results.csv not found at: {csv_path}\n"
            f"Run submit_inference.py first to generate it."
        )

    # --- Validate results.csv format ---
    with open(csv_path, "r") as f:
        lines = f.readlines()

    header = lines[0].strip()
    if header != "image_id, class_id, score":
        print(f"⚠️  Warning: CSV header is '{header}'")
        print(f"   Expected: 'image_id, class_id, score'")
    else:
        print(f"✅ CSV header format correct")

    print(f"✅ CSV contains {len(lines)-1} predictions")

    # Spot check first row
    first_row = lines[1].strip().split(",")
    print(f"✅ Sample row: image_id={first_row[0].strip()}, "
          f"class_id={first_row[1].strip()}, "
          f"score={first_row[2].strip()}")

    # --- Write readme.txt ---
    readme_content = (
        f"runtime per image [s] : {config['runtime_per_image']}\n"
        f"CPU[1] / GPU[0] : {config['uses_gpu']}\n"
        f"Extra Data [1] / No Extra Data [0] : {config['extra_data']}\n"
        f"Other description : {config['description']}\n"
    )

    with open(readme_path, "w", encoding='utf-8') as f:
        f.write(readme_content)
    print(f"\nreadme.txt written to: {readme_path}")
    print(f"\nreadme.txt contents:")
    print("-" * 50)
    print(readme_content)
    print("-" * 50)

    # --- Build ZIP (flat structure — no subfolders) ---
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path,    arcname="results.csv")
        zf.write(readme_path, arcname="readme.txt")

    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nsubmission.zip created: {zip_path}")
    print(f"ZIP size: {zip_size:.2f} MB")

    # --- Verify ZIP contents ---
    print("\nZIP contents (must be flat — no subfolders):")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename:<20} ({info.file_size / 1024:.1f} KB)")

    print("\n✅ Submission package ready.")
    print(f"   Upload this file: {zip_path}")


if __name__ == "__main__":
    build_submission(CONFIG)