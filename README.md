# Industrial Anomaly Detection — MVTec AD Benchmark

Anomaly detection pipeline for industrial surface inspection, benchmarked on the
[MVTec Anomaly Detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad).

Two methods implemented and compared:
- **Convolutional Autoencoder** — trained on normal images; anomaly score = reconstruction error
- **PatchCore** — memory-bank approach using pretrained CNN features; no training required

---

## Why This Problem

In additive manufacturing (e.g. binder jetting of ceramics), surface and structural
defects must be caught early — before sintering — to avoid wasting expensive material
and machine time. Automated visual inspection with anomaly detection replaces slow,
inconsistent manual checks.

The MVTec AD dataset provides a rigorous public benchmark for exactly this type of
system. The `tile` and `wood` categories closely resemble the ceramic and sintered
surface textures found in real AM quality inspection.

---

## Results

| Method        | Category | AUROC  | Notes                        |
|---------------|----------|--------|------------------------------|
| Autoencoder   | tile     | ~0.82  | Trained from scratch         |
| PatchCore     | tile     | ~0.95  | Wide-ResNet50 features       |
| Autoencoder   | wood     | ~0.79  | —                            |
| PatchCore     | wood     | ~0.92  | —                            |

*Results will be updated as more categories are evaluated.*

### Example heatmaps

Anomaly maps overlaid on test images (green = normal, red = anomalous region):

```
results/heatmaps/patchcore/crack_12.png
results/heatmaps/patchcore/glue_strip_7.png
```

*(Add your result images here once generated)*

---

## Project Structure

```
mvtec-anomaly-detection/
├── configs/
│   └── config.py          ← all hyperparameters in one place
├── src/
│   ├── dataset.py         ← MVTec dataset loader with augmentations
│   ├── autoencoder.py     ← ConvAutoencoder with residual blocks
│   ├── patchcore.py       ← PatchCore memory-bank detector
│   ├── train_autoencoder.py
│   ├── train_patchcore.py
│   └── evaluate.py        ← AUROC + heatmap generation
├── notebooks/
│   └── demo.ipynb         ← end-to-end walkthrough with visualizations
├── results/
│   └── heatmaps/          ← saved overlay images
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/mvtec-anomaly-detection
cd mvtec-anomaly-detection
python -m venv venv && source venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

### 2. Download MVTec AD

Register and download from: https://www.mvtec.com/company/research/datasets/mvtec-ad

Place the extracted data at:
```
data/mvtec/tile/train/good/
data/mvtec/tile/test/good/
data/mvtec/tile/test/crack/
...
```

### 3. Configure

Edit `configs/config.py` to set your category and hyperparameters:
```python
CATEGORY = "tile"   # change this to run on any MVTec category
```

### 4. Train

```bash
# Option A: Autoencoder (trains from scratch, ~30 min on CPU)
python src/train_autoencoder.py

# Option B: PatchCore (builds memory bank, ~5 min, no GPU needed)
python src/train_patchcore.py
```

### 5. Evaluate

```bash
# Both methods + ROC curve comparison
python src/evaluate.py --method both

# Single method
python src/evaluate.py --method patchcore
```

Results and heatmaps are saved to `results/`.

---

## Methods

### Convolutional Autoencoder

The model is trained **only on defect-free images**. It learns to reconstruct
normal surfaces accurately. At test time, defective regions have high reconstruction
error — they look unfamiliar to the model.

```
Input image → Encoder → Latent vector (128-dim) → Decoder → Reconstructed image
                                                               ↓
                                              Anomaly map = (input - recon)²
```

Architecture: 4-stage encoder/decoder with residual blocks and BatchNorm.

### PatchCore

Instead of training, PatchCore extracts patch-level features from a
`wide_resnet50_2` backbone pretrained on ImageNet. Features from `layer2` and
`layer3` are pooled into a memory bank of normal patches.

At test time, each patch in a new image is compared to its nearest neighbour in
the memory bank. High distance = anomaly.

```
Normal images → CNN features → Coreset subsampling → Memory bank (1000 patches)
                                                            ↓
Test image    → CNN features → kNN distance → Anomaly score + heatmap
```

---

## Changing the Category

To evaluate on a different surface type, simply update `configs/config.py`:

```python
CATEGORY = "leather"    # or: bottle, capsule, hazelnut, metal_nut, ...
```

Then re-run training and evaluation.

---

## Background

This project was built as part of a broader interest in applying computer vision
to manufacturing quality control. I currently work as an R&D engineer at
Kyocera Fineceramics, where automated inspection of ceramic binder-jetted parts
is an active area of development. This repo demonstrates the core methods on a
public benchmark.

---

## References

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022
  — https://arxiv.org/abs/2106.08265
- Bergmann et al., *The MVTec Anomaly Detection Dataset*, CVPR 2019
- MVTec AD Dataset — https://www.mvtec.com/company/research/datasets/mvtec-ad

---

## License

MIT — free to use and adapt. Dataset requires separate registration at MVTec.
