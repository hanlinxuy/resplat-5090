# 性能与数值记录

## 基线与优化顺序

主基线为 Thor 上的 ReSplat Base 初始化训练，256×448、8+6 视角、128 个深度候选、六层 Point Transformer、57,344 个高斯、BF16 主干局部 autocast、FP32 AdamW state。使用合成标定场景和官方 depth checkpoint。涉及 VGG 感知分支的性能实验使用固定种子合成权重。稳态计时排除启动和编译，CUDA 和 wall 分列。

1. **先录上游参考。** 真 B1 的 5 个稳态步中位数，官方全局 KNN **2.796 s CUDA/步**，local KNN **2.792 s**。两条路径邻域定义不同。后续融合均以 local KNN 作为相同算法的参考。[全局 KNN](../benchmarks/results/thor/full/init-official-global-v1.json) · [local KNN](../benchmarks/results/thor/full/init-local-v1.json)
2. **修复多 batch，给 KNN 候选分块。** 保留 batch 相机维度，逐场景计算，并把本地索引映射回 scene-major 点数组。B1 查询从 **46.73 ms / 1.80 GiB** 到 **32.23 ms / 142 MiB**，B1/B2/B4 与原始逐场景参考的邻居索引一一相等。此处显存是查询额外的 PyTorch 峰值分配。[索引与时延](../benchmarks/results/thor/knn-b1.json)
3. **融合候选距离和局部 attention，改变多视图卷积布局。** KNN 候选融合把查询再降到约 **13–14 ms**，沿用 Torch top-k。局部 attention 直接读取邻居 K/V 并融合 QK、softmax、PV，反向用 FP32 scatter 累加。六层子图 B1 约 **762 → 619 ms**，该 attention 样本的输出、输入梯度、最大参数梯度相对 L2 分别为 **0.477%、0.536%、0.856%**。多视图卷积 opt-in channels-last。三项一起打开，B1 整步 **2.792 → 2.610 s CUDA**，峰值分配 **28.49 → 27.08 GiB**，同窗口逐步 loss 最大差约 **4.83e-4**。[KNN 融合](../benchmarks/results/thor/knn-fused-b1.json) · [attention 数值](../benchmarks/results/thor/fused-local-r4-repeat-b1.json) · [整步融合](../benchmarks/results/thor/full/init-optimized-v1.json)
4. **单独验证感知和更新路径。** VGG 卷积和池化走 BF16 autocast，feature 差分及 loss 归约保留 FP32。固定其余布局的独立 A/B 中，CUDA 整步 **2.667 → 1.432 s**，对齐样本的最大 loss 差 **1.51e-4**。7 次更新后三个抽样张量最大参数差 **0.084% 相对 L2**，最大梯度差 **4.32%**。独立 fused AdamW 将更新段约 **80 → 35 ms**，保持 FP32 state，4 次更新后抽样参数最大相对 L2 差 **0.065%**。它们是独立窗口，整步收益不能直接相加。[VGG FP32](../benchmarks/results/thor/full/init-vgg-precision-fp32.json) · [VGG BF16](../benchmarks/results/thor/full/init-vgg-precision-bf16.json) · [VGG 数值](../benchmarks/results/thor/full/validation-vgg-bf16.json) · [AdamW 数值](../benchmarks/results/thor/full/validation-fused-adamw.json)
5. **组合到训练入口。** 有效 B32 的数据及 context 视角逐项相同。local KNN、三项融合、组合版的 wall 周期为 **70.33 → 61.84 → 49.04 s**。组合版与 local KNN 的 16 个采样点最大 loss 差 **9.92e-6**。这是一次有效 B32 更新，B1 稳态另有独立证据。[原始周期](../benchmarks/results/thor/full/init-32scenes-local.json) · [三项融合](../benchmarks/results/thor/full/init-32scenes-optimized.json) · [组合版](../benchmarks/results/thor/full/init-32scenes-mixed.json)

用同一配方重复跑一次也会观察到差异。Thor local KNN 重复对照在 4 次更新后的三个抽样参数中，最大相对 L2 为 **0.061%**，三项融合版对应 **0.099%**。参数与优化器 audit 每张量抽样 256 个元素。另一次 Nsight Systems 对照显示，每步保留的 CUDA 核从 **20,789 降到 6,626**，Tensor Active 从 **4.22% 到 8.21%**，均为该采样窗口。[重复对照](../benchmarks/results/thor/full/validation-local-repeat.json) · [融合对照](../benchmarks/results/thor/full/validation-optimized.json) · [基线 profile](../benchmarks/results/thor/full/full-init-local-marked-summary.json) · [组合版 profile](../benchmarks/results/thor/full/full-init-mixed-marked-summary.json)

## 其他独立负载

| 负载 | 已测结果 | 数值范围 |
| --- | --- | --- |
| Thor refinement，1/4/2 次迭代 | CUDA 三步 **13.197 → 4.787 s**，按 1、4、2 次迭代逐步对齐，峰值 **33.45 → 23.88 GiB** | 最大 loss 绝对差 **0.00757**，抽样末步参数差 **0.0161% 相对 L2** |
| 256×448，有效 B32，随机 depth 权重 | 3090 **23.68 s**、4090 **15.78 s** wall | 同一批 32 个合成场景、同机位 |
| 4090，原生 800×800，6 层 B1 | CUDA **2.635 s/步**，wall **2.748 s/步**，整卡采样峰值 **21.92 GiB** | D8 融合 cost volume，D=19 回归的输出与两路特征梯度相对 L2 为 **1.49e-7 / 8.47e-8 / 1.61e-7** |
| 3090，448×224，50,176 → 802,816 高斯 | 8+6 视角，wall **645 → 803 ms/步**，其中反向 **410 → 548 ms** | 80 万版追加 2,160 步后，4 个留出合成人物前景 PSNR **23.18 dB**，50K +216 步对照为 **23.88 dB**。两路步数及密度配方不同 |

[Refinement 数值](../benchmarks/results/thor/full/validation-refine-mixed.json) · [3090/4090 对照](../benchmarks/results/) · [800×800 与算子测试](../benchmarks/results/rtx4090/native800/) · [80 万高斯长训](../benchmarks/results/rtx3090/face-cafca-448-803k-extended/summary.json)

原生 800×800 用 D=8 分块融合 warp 与 correlation，反向逐组用 FP32 grid sampler 重算。六层还开启 depth、Point Transformer 和感知分支重算。分辨率、层数、高斯密度、loss 或重算配置变化时，独立报告吞吐与质量。精度验收和 5090 上的复现步骤见 [Handoff](HANDOFF.md)。
