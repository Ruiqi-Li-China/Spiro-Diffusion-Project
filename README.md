# 🫁 Spiro-Diffusion: Multimodal Conditioned Lung Function Reconstruction
## 📖 Overview
**Spiro-Diffusion** is a deep learning framework designed to reconstruct and generate high-fidelity pulmonary Flow-Volume loops (Spirometry curves).
Unlike traditional interpolation methods, this project utilizes a **Physics-Guided Conditional Latent Diffusion Model (cLDM)**. It generates 1D signals conditioned on patient-specific clinical phenotypes (**Age, Height, Gender**) to ensure physiological consistency.

This research uses data from the **NHANES 2011-2012 (Cycle G)** dataset.
---

## 🏗️ Technical Architecture

The project follows a two-stage generative pipeline:

1. **Phase 1: Latent Representation Learning (VQ-VAE)**
* Compresses high-dimensional 1D spirometry signals () into a discrete latent space ().
* **Architecture:** 1D Convolutional Encoder-Decoder with Vector Quantization.
  
2. **Phase 2: Conditional Latent Diffusion (cLDM)**
* Generates latent representations from Gaussian noise.
* **Guidance Mechanism:** A Conditional U-Net that injects clinical metadata (Age, Height, Gender) via Cross-Attention layers.
---

## 📂 Project Structure

```bash
Spiro-Diffusion-Project/
├── data/
│   ├── raw/nhanes/           # Place raw SAS/XPT files here (SPXRAW_G, DEMO_G, BMX_G)
│   └── processed/            # Generated .npy and .csv files
├── src/
│   ├── models/
│   │   ├── vq_vae.py         # VQ-VAE Model Architecture
│   │   └── diffusion_unet.py # Conditional U-Net Architecture
│   ├── preprocess_multimodal.py  # Data Alignment & Resampling
│   ├── train_vqvae.py        # Phase 1 Training Script
│   ├── prepare_latents.py    # Pre-calculates latents for Phase 2
│   ├── train_cldm.py         # Phase 2 (Diffusion) Training Script
│   └── inference_cldm.py     # Generate synthetic curves
├── checkpoints/              # Saved model weights (.pth)
├── DEV_LOG.md                # Development Diary
└── README.md                 # Project Documentation

```

---

## 🚀 Getting Started

### 1. Prerequisites

Install the required dependencies:

```bash
pip install torch torchvision numpy pandas scipy matplotlib

```

### 2. Data Preparation

Due to GitHub size limits, raw data is not included.

1. **Download** the following files from the [CDC NHANES 2011-2012 Website](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011):
* `SPXRAW_G.sas7bdat` (Spirometry - Raw Curve)
* `DEMO_G.xpt` (Demographics)
* `BMX_G.xpt` (Body Measures)


2. **Place them** in `data/raw/nhanes/`.
3. **Run Preprocessing:**
```bash
python src/preprocess_multimodal.py

```


*Output:* `data/processed/signals_L512.npy` and `metadata_aligned.csv`.

---

## 🏃‍♂️ Training Pipeline

### Phase 1: VQ-VAE (Compression)

Train the autoencoder to learn how to compress and reconstruct flow-volume loops.

```bash
python src/train_vqvae.py

```

### Phase 2: Latent Diffusion (Generation)

**Step A: Pre-calculate Latents**
To speed up training, we pre-encode the entire dataset into latent vectors.
*(Note: This generates `latents.npy`, which is ~1.2GB and excluded from git)*

```bash
python src/prepare_latents.py

```

**Step B: Train the Diffusion Model**
Trains the cLDM to denoise signals based on Age/Height/Gender.

```bash
python src/train_cldm.py

```

---

## 🧪 Inference (Testing)

To generate a synthetic lung function curve for a specific "Digital Patient":

```bash
python src/inference_cldm.py

```

*You can modify the `age`, `height`, and `gender` parameters inside the script to test different patient profiles.*

---

## 📊 Results & Visualization

| Input Condition | Generated Output |
| --- | --- |
| **Male, 45y, 175cm** | *(See generated_result.png)* |

*(The model learns to generate the characteristic rapid peak flow and linear expiration decline typical of healthy lung function.)*

---

## 📝 License

This project is for research purposes. Data usage must comply with CDC NHANES guidelines.

**Maintainer:** [Ruiqi Li]
**Last Updated:** Feb 2026

# 🫁 Spiro-Diffusion：多模态条件肺功能重建

## 📖 概述

**Spiro-Diffusion** 是一个深度学习框架，旨在重建和生成高保真度的肺部用力呼气量-容积环（肺功能曲线）。

与传统的插值方法不同，本项目采用了一种**物理引导的条件潜在扩散模型 (cLDM)** 进行生成。该模型可以根据患者的临床特征（**年龄、身高、性别**）来生成 1D 信号，从而保证生理上的合理性。

本研究使用的是**NHANES 2011-2012（Cycle G）**数据集。

---

## 🏗️ 技术架构

本项目采用了两阶段的生成流程：

1. **阶段 1：潜在表示学习（VQ-VAE）**
   - 将高维度的 1D 肺功能信号压缩到离散的潜在空间中。
   - **架构**：使用 1D 卷积编码器-解码器结构，结合向量量化（Vector Quantization）。

2. **阶段 2：条件潜在扩散模型（cLDM）**
   - 从高斯噪声中生成潜在表示。
   - **引导机制**：使用条件 U-Net，通过交叉注意力（Cross-Attention）层注入临床元数据（年龄、身高、性别）。

---

## 📂 项目结构

```
Spiro-Diffusion-Project/
├── data/
│   ├── raw/nhanes/           # 放置原始 SAS/XPT 文件（SPXRAW_G、DEMO_G、BMX_G）
│   └── processed/            # 生成的 .npy 和 .csv 文件
├── src/
│   ├── models/
│   │   ├── vq_vae.py         # VQ-VAE 模型架构
│   │   └── diffusion_unet.py # 条件 U-Net 模型架构
│   ├── preprocess_multimodal.py  # 数据对齐与重采样
│   ├── train_vqvae.py        # 阶段 1 训练脚本
│   ├── prepare_latents.py    # 预先计算潜在表示，用于阶段 2
│   ├── train_cldm.py         # 阶段 2（扩散）训练脚本
│   └── inference_cldm.py     # 生成合成曲线
├── checkpoints/              # 保存模型权重（.pth 文件）
├── DEV_LOG.md                # 项目开发日志
└── README.md                 # 项目文档
```

---

## 🚀 开始使用

### 1. 依赖项安装

请先安装所需的库：

```bash
pip install torch torchvision numpy pandas scipy matplotlib
```

### 2. 数据准备

由于 GitHub 的大小限制，原始数据文件**不会包含在仓库中**。

1. **从以下链接下载** 文件：
   [CDC NHANES 2011-2012 网站](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011)
   - `SPXRAW_G.sas7bdat`（肺功能 - 原始曲线）
   - `DEMO_G.xpt`（人口统计数据）
   - `BMX_G.xpt`（身体测量）

2. **将文件放入** `data/raw/nhanes/` 文件夹中。
3. **运行预处理脚本**：

   ```bash
   python src/preprocess_multimodal.py
   ```

   *输出：* 生成 `data/processed/signals_L512.npy` 和 `metadata_aligned.csv`。

---

## 🏃‍♂️ 训练流程

### 阶段 1：VQ-VAE（压缩）

训练自动编码器，用于学习如何压缩和重建肺功能曲线。

```bash
python src/train_vqvae.py
```

### 阶段 2：潜在扩散模型（生成）

#### **步骤 A：预先计算潜在表示**
为了加快训练速度，我们预先对整个数据集进行编码，转换为潜在向量。
（注意：这会生成 `latents.npy` 文件，约 1.2GB，**不在 Git 中存储**）

```bash
python src/prepare_latents.py
```

#### **步骤 B：训练扩散模型**
训练 cLDM 模型，根据年龄、身高、性别等信息去噪信号。

```bash
python src/train_cldm.py
```

---

## 🧪 推理（测试）

要为特定的“数字患者”生成合成肺功能曲线：

```bash
python src/inference_cldm.py
```

*可以**修改脚本中**的 `age`、`height` 和 `gender` 参数，以测试不同的患者配置。*

---

## 📊 结果与可视化

| 输入条件 | 生成结果 |
|----------|----------|
| **男性，45 岁，身高 175cm** | *(查看生成结果：generated_result.png)* |

（模型能够生成具有健康肺功能特征的典型快速峰值流和线性呼气衰减曲线。）

---

## 📝 许可证

本项目仅供研究使用。数据使用需遵守 CDC NHANES 指南。

**维护人：** [李睿琪]
**最后更新：** 2026 年 2 月

