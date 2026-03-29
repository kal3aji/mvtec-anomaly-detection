"""
Central configuration — change values here, everything else picks them up.
"""

# ── Dataset ───────────────────────────────────────────────────────────────────
CATEGORY = "tile"          # MVTec category to train/eval on
                           # Options: tile, wood, leather, hazelnut, metal_nut,
                           #          bottle, capsule, carpet, grid, pill,
                           #          screw, toothbrush, transistor, zipper
DATA_ROOT = "data/mvtec"
IMAGE_SIZE = 256           # resize to this before processing

# ── Training (Autoencoder) ────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-3
LATENT_DIM    = 128        # bottleneck size

# ── PatchCore ─────────────────────────────────────────────────────────────────
PATCHCORE_BACKBONE   = "wide_resnet50_2"   # or "resnet18" for faster runs
PATCHCORE_LAYERS     = ["layer2", "layer3"]
PATCHCORE_CORESET_N  = 1000               # memory bank size after subsampling
PATCHCORE_PATCH_SIZE = 3                  # neighbourhood for patch descriptors

# ── Inference ─────────────────────────────────────────────────────────────────
ANOMALY_THRESHOLD = 0.5    # tuned per category during evaluation
HEATMAP_ALPHA     = 0.4    # overlay transparency

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = "models"
RESULTS_DIR = "results"
