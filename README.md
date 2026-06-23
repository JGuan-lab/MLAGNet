# MLAGNet

**English** | [中文](#中文)

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

---
---

# 中文

[English](#mlagnet) | **中文**

> 面向医学影像的**统一特征提取框架**，用同一套核心模块（深度可分离卷积 + 局部自注意力 + 门控前馈）支撑**超分辨率、语义分割、图像分类**三类任务。

本仓库为毕业论文代码整理版。三个任务复用相同的特征提取设计（`DConv` + `MDTA/LSA` + `GDFN` + `Transformerblock`），分别构建为：

| 任务 | 网络范式 | 主网络文件 |
|------|----------|-----------|
| 超分辨率 | GAN（编码-解码生成器 + UNet 判别器） | `superresolution/Gen.py` |
| 语义分割 | UNet 变体（编码-解码） | `segmentation/uufenge.py` |
| 图像分类 | Encoder-only + MLP 头 | `classification/uuclassifier.py` |

> 说明：三个任务在**架构设计上统一**，但各自独立实现核心模块（并非共享同一份权重/同一个 Python 包）。

---

## 目录结构

```
MLAGNet/
├── superresolution/      # 超分辨率（肺部 CT，2× / 4× / 8×）
│   ├── Gen.py                #  ★ 生成器（编码-解码 + 多尺度加权聚合 + PixelShuffle 上采样）
│   ├── UnetDiscriminator.py  #  UNet 判别器（谱归一化，Real-ESRGAN 风格）
│   ├── waveloss.py           #  ★ 小波(SWT)损失 + 相对 GAN 损失 + VGG 感知损失
│   ├── metric.py             #  PSNR / SSIM
│   ├── degradtion.py         #  Real-ESRGAN 二阶退化（依赖 basicsr；默认未接入训练）
│   ├── trainVAL.py           #  ★ 训练入口（4×，小波 GAN，accelerate，逐 epoch 验证）
│   ├── trainNoWave.py        #  训练入口（4×，标准 GAN：L1+VGG感知+相对GAN，支持断点续训）
│   ├── testfull.py / testfull2.py / testfull8.py  #  ★ 批量测试 4× / 2× / 8×
│   ├── testone.py            #  单图测试
│   ├── config.py             #  ★ 路径集中配置
│   ├── Ui/                   #  Gradio 交互式演示（自包含）
│   └── check/                #  实验性脚本（自包含）
│
├── classification/       # 图像分类（4 个医学数据集）
│   ├── uuclassifier.py       #  ★ 自研分类网络（Encoder + MLP 头）
│   ├── trainisic.py / trainmhist.py / trainh.py / trainx.py  #  ★ 各数据集训练
│   ├── test.py               #  测试 + 混淆矩阵
│   ├── redraw_vit_eval.py    #  ViT 基线评估（重绘混淆矩阵/ROC）
│   ├── config.py             #  ★ 路径集中配置
│   └── xiaorong/             #  消融：NoGDFN / NoLSA
│
└── segmentation/         # 语义分割（3 个医学数据集）
    ├── uufenge.py            #  ★ 自研分割网络（UNet 变体，各阶段 Transformer block 数可配）
    ├── trainani.py / traincrack.py / trainretina.py  #  ★ 训练：血管造影 / 裂缝 / 视网膜
    ├── train0/1/4/11/22/44.py, trainup1/2/4.py  # block 数量消融（命名 XY = up 块数/down 块数）
    ├── test.py / test1.py    #  测试（视网膜 / 裂缝）
    ├── config.py             #  ★ 路径集中配置
    └── xiaorong/             #  消融：full / NoGDFN / NoLSA
```

> 标 ★ 的为规范入口/核心文件；其余 `train*` / `test*` 变体为不同倍率、不同数据集或消融对照。

---

## 核心模块（三任务共享的设计）

- **`DConv`** — 深度可分离卷积（depthwise 3×3 + pointwise 1×1 + BN + 激活），全框架基础构件。
- **`MDTA` / `LSA`** — 局部自注意力：以 `DConv` 构建 Q/K/V；浅层做空间注意力、深层做通道注意力，带可学习残差系数。
- **`GDFN`** — 门控双分支前馈网络：两路展宽 → DConv → SiLU 门控相乘 → 压缩 → 残差。
- **`Transformerblock`** — `MDTA` + `GDFN` 串联，是编码/解码各阶段的基本单元。
- **`WeightedFeatureAggregation`** — 多尺度特征的可学习加权融合。
- **`CBAM`** — 通道+空间注意力（用于超分上采样块）。

超分与分割为**编码-解码**结构；分类为**编码器 + 全局池化 + MLP** 结构。

---

## 环境依赖

```bash
# 核心
torch  torchvision  accelerate
numpy  opencv-python  Pillow  matplotlib  tqdm  scipy  scikit-image  thop

# 超分小波损失（按 waveloss.py 实际实现二选一）
pytorch_wavelets   # 或  PyWavelets(pywt)

# 超分退化合成（仅 degradtion.py 需要）
basicsr

# 超分 Gradio 演示（仅 superresolution/Ui 需要）
gradio
```

建议 Python 3.10 + CUDA（实验环境为 2× RTX 3090）。多卡训练通过 `accelerate` 启动：

```bash
accelerate launch superresolution/trainVAL.py
```

---

## 路径配置

所有数据集/权重/输出路径集中在各任务模块下的 `config.py`（`superresolution/`、`classification/`、`segmentation/`）。适配新环境二选一——设环境变量（推荐，不改代码）：

```bash
export MLAG_DATA_ROOT=/your/data/root      # 数据集根目录
export MLAG_SR_OUTPUT=/your/sr/model/dir   # 超分权重输出（可选）
export MLAG_CLS_MODEL=/your/cls/model/dir  # 分类权重根（可选）
export MLAG_SEG_MODEL=/your/seg/model/dir  # 分割权重根（可选）
```

……或直接改对应模块 `config.py` 顶部默认值。默认值为原实验环境路径，日志/结果默认写到各模块目录内。

---

## 快速开始

```bash
# 超分辨率
accelerate launch superresolution/trainVAL.py   # 训练（4×，小波 GAN）
python superresolution/testfull.py              # 测试（testfull=4× / testfull2=2× / testfull8=8×）
python superresolution/testone.py               # 单图推理

# 分类
python classification/trainisic.py   # ISIC / trainmhist.py / trainh.py / trainx.py
python classification/test.py        # 评估 + 混淆矩阵

# 分割
python segmentation/trainani.py      # 血管造影 / traincrack.py / trainretina.py
python segmentation/test.py          # 测试
```

---

## 数据集

| 任务 | 数据集 | 类型 |
|------|--------|------|
| 超分 | 肺部 CT、Kaggle COVID CT、裁剪 CT | 配对 GT/LR（2×/4×/8× 下采样） |
| 分类 | ISIC（皮肤镜）、MHIST（组织病理）、Histopathology（乳腺）、ChestX-Ray（胸片） | `ImageFolder` 目录式 |
| 分割 | 血管造影、裂缝、视网膜血管 | 图像 + 掩码（labelcol） |

> ⚠️ 公开仓库**只含代码、不含数据**。请自行准备数据集并让 `config.py` / `MLAG_DATA_ROOT` 指向它。

### 数据获取（Zenodo）

完整数据集已归档到 Zenodo：**DOI [10.5281/zenodo.20807732](https://doi.org/10.5281/zenodo.20807732)**（concept DOI，永久指向最新版）— 记录页：<https://zenodo.org/record/20807733>。

| 压缩包 | 内容 |
|--------|------|
| `superresolution.tar` | 肺部 CT（高清 + 降质）、裁剪 CT、Kaggle COVID CT（多倍率） |
| `segmentation.tar` | 血管造影 / 裂缝 / 视网膜血管（图像 + 掩码） |
| `classification.tar.part00 … part08` | MHIST / 胸片 / ISIC，切成 2GB 分块 |

分类包因 Zenodo 不接受超大单文件而切块，下载后这样合并：

```bash
cat classification.tar.part* > classification.tar && tar -xf classification.tar
```

> 乳腺**组织病理（histopathology）**分类数据集因体量过大**未放入** Zenodo，请从 [Hugging Face](https://huggingface.co/datasets/EulerianKnight/breast-histopathology-images-train-test-valid-split) 获取。

---

## 注意事项

1. **重复脚本**：超分 `testfull/testfull2/testfull8` = 4×/2×/8×（仅 `num_upsample` 与路径不同）；分割 `train0/1/4/11/22/44 / trainup*` 是同一血管造影数据集上对 Transformer block 数量的消融（主力为 `trainani.py`）；分类 `train{isic,mhist,h,x}` = 同一网络在 4 个数据集上训练。
2. **训练用预生成 LR**：超分训练读取预先下采样的 LR 图像对；`degradtion.py`（在线 Real-ESRGAN 退化）作为可选方案保留，默认未接入训练循环。
3. **已知小问题**（整理时发现，保持原貌未改）：`superresolution/waveloss.py` 的 `DLoss` 高频(HL/HH)分支疑似把 fake 与 fake 比较；`segmentation/uufenge.py` 的多尺度融合结果未被最终 logits 使用；`segmentation/test1.py` 的 `Classifier(2)` 缺少 `upblocks/downblocks` 参数。

---

## 说明

本仓库由原始论文工程 `uureal` 整理而来：仅保留代码（去除数据、权重、日志、输出图与缓存），按 **超分 / 分割 / 分类** 三任务重新归类，并清理了少量损坏/空文件。原始完整工程（含数据与权重）保留在实验室服务器，未做改动。
