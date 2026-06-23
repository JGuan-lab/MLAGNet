# MLAGNet

> A **unified feature-extraction framework for medical imaging** that drives three tasks — **super-resolution, semantic segmentation, and image classification** — with the same set of core building blocks (depthwise separable convolution + local self-attention + gated feed-forward).

This repository is the cleaned-up version of the thesis codebase. The three tasks share the same feature-extraction design (`DConv` + `MDTA/LSA` + `GDFN` + `Transformerblock`) and are built as:

| Task | Paradigm | Main network |
|------|----------|--------------|
| Super-Resolution | GAN (encoder–decoder generator + UNet discriminator) | `superresolution/Gen.py` |
| Segmentation | UNet variant (encoder–decoder) | `segmentation/uufenge.py` |
| Classification | Encoder-only + MLP head | `classification/uuclassifier.py` |

> Note: the three tasks are **unified in architectural design**, but each implements the core modules independently (they do **not** share one Python package or one set of weights).

---

## Directory Structure

```
MLAGNet/
├── superresolution/      # Super-resolution (lung CT, 2x / 4x / 8x)
│   ├── Gen.py                #  ★ Generator (encoder-decoder + multi-scale weighted aggregation + PixelShuffle)
│   ├── UnetDiscriminator.py  #  UNet discriminator (spectral norm, Real-ESRGAN style)
│   ├── waveloss.py           #  ★ Wavelet (SWT) loss + relativistic GAN loss + VGG perceptual loss
│   ├── metric.py             #  PSNR / SSIM
│   ├── degradtion.py         #  Real-ESRGAN 2nd-order degradation (needs basicsr; not wired into training by default)
│   ├── trainVAL.py           #  ★ Training entry (4x, wavelet GAN, HuggingFace accelerate, per-epoch validation)
│   ├── trainNoWave.py        #  Training entry (4x, standard GAN: L1+VGG+relativistic, supports resume)
│   ├── testfull.py / testfull2.py / testfull8.py  #  ★ Batch test for 4x / 2x / 8x
│   ├── testone.py            #  Single-image test
│   ├── config.py             #  ★ Centralized paths
│   ├── Ui/                   #  Gradio interactive demo (self-contained)
│   └── check/                #  Experimental scripts (self-contained)
│
├── classification/       # Classification (4 medical datasets)
│   ├── uuclassifier.py       #  ★ Classifier network (Encoder + MLP head)
│   ├── trainisic.py / trainmhist.py / trainh.py / trainx.py  #  ★ Training per dataset
│   ├── test.py               #  Test + confusion matrix
│   ├── redraw_vit_eval.py    #  ViT baseline evaluation (redraw confusion matrix / ROC)
│   ├── config.py             #  ★ Centralized paths
│   └── xiaorong/             #  Ablations: NoGDFN / NoLSA
│
└── segmentation/         # Segmentation (3 medical datasets)
    ├── uufenge.py            #  ★ Segmentation network (UNet variant, configurable #Transformer blocks per stage)
    ├── trainani.py / traincrack.py / trainretina.py  #  ★ Training: angiography / crack / retina
    ├── train0/1/4/11/22/44.py, trainup1/2/4.py  # Block-count ablations (naming XY = #up / #down blocks)
    ├── test.py / test1.py    #  Test (retina / crack)
    ├── config.py             #  ★ Centralized paths
    └── xiaorong/             #  Ablations: full / NoGDFN / NoLSA
```

> ★ marks canonical entry points / core files; other `train*` / `test*` variants are different scales, datasets, or ablation runs.

---

## Core Modules (shared across all tasks)

- **`DConv`** — depthwise separable convolution (depthwise 3×3 + pointwise 1×1 + BN + activation); the basic building block.
- **`MDTA` / `LSA`** — local self-attention: Q/K/V built with `DConv`; spatial attention at shallow stages, channel attention at deep stages, with a learnable residual coefficient.
- **`GDFN`** — gated dual-branch feed-forward: two branches expand → DConv → SiLU gating → compress → residual.
- **`Transformerblock`** — `MDTA` + `GDFN` in series; the basic unit of each encoder/decoder stage.
- **`WeightedFeatureAggregation`** — learnable weighted fusion of multi-scale features.
- **`CBAM`** — channel + spatial attention (used in SR upsampling blocks).

Super-resolution and segmentation use an **encoder–decoder**; classification uses an **encoder + global pooling + MLP head**.

---

## Requirements

```bash
# core
torch  torchvision  accelerate
numpy  opencv-python  Pillow  matplotlib  tqdm  scipy  scikit-image  thop

# SR wavelet loss (depending on waveloss.py implementation)
pytorch_wavelets   # or  PyWavelets (pywt)

# SR degradation synthesis (only for degradtion.py)
basicsr

# SR Gradio demo (only for superresolution/Ui)
gradio
```

Python 3.10 + CUDA recommended (experiments used 2× RTX 3090). Multi-GPU training is launched via `accelerate`:

```bash
accelerate launch superresolution/trainVAL.py
```

---

## Configuration

All dataset / weight / output paths are centralized in a `config.py` inside each task module (`superresolution/`, `classification/`, `segmentation/`). To adapt to a new environment, either set environment variables (recommended, no code change):

```bash
export MLAG_DATA_ROOT=/your/data/root      # dataset root
export MLAG_SR_OUTPUT=/your/sr/model/dir   # SR weights output (optional)
export MLAG_CLS_MODEL=/your/cls/model/dir  # classification weights root (optional)
export MLAG_SEG_MODEL=/your/seg/model/dir  # segmentation weights root (optional)
```

…or edit the defaults at the top of each module's `config.py`. Defaults reproduce the original experiment paths; logs/results are written inside each module directory by default.

---

## Quick Start

```bash
# Super-resolution
accelerate launch superresolution/trainVAL.py   # train (4x, wavelet GAN)
python superresolution/testfull.py              # test (testfull=4x / testfull2=2x / testfull8=8x)
python superresolution/testone.py               # single-image inference

# Classification
python classification/trainisic.py   # ISIC / trainmhist.py / trainh.py / trainx.py
python classification/test.py        # evaluate + confusion matrix

# Segmentation
python segmentation/trainani.py      # angiography / traincrack.py / trainretina.py
python segmentation/test.py          # test
```

---

## Datasets

| Task | Datasets | Format |
|------|----------|--------|
| SR | Lung CT, Kaggle COVID CT, cropped CT | paired GT/LR (2x/4x/8x downsampling) |
| Classification | ISIC (dermoscopy), MHIST (histopathology), Histopathology (breast), ChestX-Ray | `ImageFolder` directory-based |
| Segmentation | Angiography, Crack, Retina vessels | image + mask (labelcol) |

> ⚠️ The public repository contains **code only, no data**. Prepare datasets yourself and point `config.py` / `MLAG_DATA_ROOT` to them.

### Data availability (Zenodo)

The full datasets are archived on Zenodo: **DOI [10.5281/zenodo.20807732](https://doi.org/10.5281/zenodo.20807732)** (concept DOI, always resolves to the latest version) — record: <https://zenodo.org/record/20807733>.

| Archive | Content |
|---------|---------|
| `superresolution.tar` | lung CT (HR + degraded), cropped CT, Kaggle COVID CT (multi-scale) |
| `segmentation.tar` | angiography / crack / retinal-vessel (image + mask) |
| `classification.tar.part00 … part08` | MHIST / Chest X-Ray / ISIC, split into 2 GB parts |

The classification archive is split because Zenodo rejects very large single files. Reassemble it with:

```bash
cat classification.tar.part* > classification.tar && tar -xf classification.tar
```

> The breast **histopathology** classification dataset is **not** in the Zenodo archive due to size; get it from [Hugging Face](https://huggingface.co/datasets/EulerianKnight/breast-histopathology-images-train-test-valid-split).

---

## Notes

1. **Duplicate scripts**: SR `testfull/testfull2/testfull8` = 4x/2x/8x (only the `num_upsample` and paths differ); segmentation `train0/1/4/11/22/44 / trainup*` are Transformer-block-count ablations on the same angiography dataset (canonical = `trainani.py`); classification `train{isic,mhist,h,x}` = the same network on 4 different datasets.
2. **Training uses pre-generated LR**: SR training reads pre-downsampled LR pairs; `degradtion.py` (online Real-ESRGAN degradation) is kept as an optional path, not wired into the training loop by default.
3. **Known minor issues** (found during cleanup, left as-is to preserve the original): the HL/HH branch in `superresolution/waveloss.py` `DLoss` appears to compare fake vs fake; `WeightedFeatureAggregation` output in `segmentation/uufenge.py` is computed but not used by the final logits; `segmentation/test1.py` `Classifier(2)` is missing the `upblocks/downblocks` args.

---

## About

This repository was organized from the original thesis project `uureal`: code only (data, weights, logs, output images, and caches removed), reorganized into **super-resolution / segmentation / classification**, with a small number of broken/empty files cleaned up. The full original project (with data and weights) is kept on the lab server, unchanged.




