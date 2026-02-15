🫁 项目开发日志 (DEV_LOG)
📝 项目信息
项目名称：Spiro-Diffusion（多模态肺功能曲线重建）
维护者：[Ruiqi-Li-China]
状态：第二阶段（扩散模型训练）已完成
最后更新：2026-02-16
📅 已完成的工作总结
✅ Phase 1: 数据工程与多模态对齐
目标：构建一个高质量的数据集，将肺功能曲线与临床特征（如年龄、身高、性别等）关联。
数据来源：NHANES 2011-2012（Cycle G）
执行操作：
下载了 SPXRAW_G（原始曲线）、DEMO_G（人口统计数据）、BMX_G（身体测量数据）
通过手动验证解决了 XPT 文件头损坏的问题
实现了 src/preprocess_multimodal.py，通过 SEQN 字段进行内连接（INNER JOIN）
对所有曲线进行重采样，统一长度为 512 点（单位：L/s）
成果：
成功对齐了 36,873 位 患者数据
生成了 metadata_aligned.csv（包含年龄、身高、性别等临床特征）
生成了 signals_L512.npy（标准化的 1D 曲线数据）
✅ Phase 1.2: 潜在表示学习（VQ-VAE）
目标：将高维信号压缩为离散的潜在代码（latent codes）
模型：1D VQ-VAE（编码器 → 量化器 → 解码器）
训练：
输入为 signals_L512.npy
压缩比：512 个点 → 64 个潜在令牌（latent tokens）
使用脚本：src/train_vqvae.py
成果：
模型训练完成并保存为 checkpoints/vqvae_phase1.pth
✅ Phase 2: 条件潜在扩散模型（cLDM）
目标：根据患者的生物特征（如年龄、身高、性别）生成肺功能曲线
潜在表示生成：
通过运行 src/prepare_latents.py，将所有 36,000 条信号转换为潜在向量（64 × 128）
保存为 latents.npy（注意：约 1.2GB，git 不会上传）
模型：基于临床特征的 1D 条件 U-Net（使用时序嵌入与时序交叉注意力）
训练：
使用脚本 src/train_cldm.py
通过过滤缺失身高数据的 220 行解决了 NaN Loss 的问题
损失值从 1.07 降低到约 0.15
成果：
模型保存为 checkpoints/cldm_phase2.pth
📂 数据集清单
为确保结果可复现，data 文件夹应包含以下文件：

文件名	描述	当前仓库状态
metadata_aligned.csv	对齐后的临床数据（年龄、身高、性别等）	✅ 已上传
signals_L512.npy	预处理后的 1D 曲线（长度为 512）	✅ 已上传
SPX_G.xpt	原始肺功能数据（未压缩）	✅ 已上传
latents.npy	VQ-VAE 编码后的潜在表示	❌ 过大（本地生成）
🚀 如何继续工作？
如果丢失了 latents.npy，可以通过以下命令重新生成：

从 signals_L512.npy 文件中生成潜在表示：

<BASH>
python src/prepare_latents.py
继续训练 cLDM 模型：

<BASH>
python src/train_cldm.py
进行推理（生成合成肺功能曲线）：

<BASH>
python src/inference_cldm.py
✅ 您可以在脚本中修改 age、height 和 gender 参数，以测试不同的患者配置

📊 结果与可视化
输入条件	生成结果
男性，45 岁，身高 175cm	查看生成结果：generated_result.png
（模型会生成具有健康肺功能特征的典型快速峰值流和线性呼气衰减曲线。）

📝 许可证
本项目仅供研究使用。数据使用需遵守 CDC NHANES 指南。

📌 总结
本项目使用 NHANES 2011-2012 数据集进行肺功能曲线的重建和生成。
项目分为两个主要阶段：
Phase 1：使用 VQ-VAE 压缩肺功能数据
Phase 2：使用 cLDM 模型根据患者特征生成曲线
所有数据文件已上传到 GitHub，大小均在限制范围内
latents.npy 文件体积过大，建议在本地生成

&nbsp;  python src/prepare\_latents.py




