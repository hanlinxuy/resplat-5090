# ReSplat 5090

ReSplat Base 的单卡训练优化与复现仓库。保留上游初始化和递归 refinement 入口，加入多 batch local KNN、Triton 局部注意力、卷积布局优化、分块 cost volume 和整步 profiling。自写 GPU 路径采用通用 Triton，依赖库按正常规则选择内核。RTX 5090 是下一台验收设备，已有实测来自 Jetson Thor、RTX 3090 和 RTX 4090。

## 原始速度与加速过程

下表是 Thor、ReSplat Base 初始化、256×448、8 输入 + 6 监督视角、各 128 个深度候选的有效 B32 实测。两边均为真 B2 × 16 累积、同顺序的 32 个合成场景、一次 AdamW 更新。初始化、局部算子和 refinement 的独立基线见[实验记录](docs/BENCHMARKS.md)。

| 实测阶段 | CUDA 更新周期 | Wall 更新周期 | Wall 吞吐 | 峰值分配 |
| --- | ---: | ---: | ---: | ---: |
| 原 local KNN 训练基线 | 66.78 s | 70.33 s | 0.455 场景/s | 42.90 GiB |
| KNN 候选融合 + 局部 attention 融合 + 多视图卷积 channels-last | 58.34 s | 61.84 s | 0.517 场景/s | 40.07 GiB |
| 再开 VGG BF16 与 fused AdamW | **45.68 s** | **49.04 s** | **0.653 场景/s** | **31.54 GiB** |

组合版 wall 吞吐提升 **1.43×**，相同场景和机位的最大逐微步 loss 绝对差 **9.92e-6**。局部 KNN 在 B1/B2/B4 的邻居索引全部一致，融合 attention、VGG 低精度和 AdamW 各有输出或梯度对照。上游全局 KNN 和 local KNN 采用不同邻域，分别记录。合成短程实验只验证对应 fixture 的训练行为，BF16 感知分支仍需用正式感知权重、真实验证集和长程训练验收。[逐步收益与数值证据](docs/BENCHMARKS.md)

## 使用

- [Handoff](docs/HANDOFF.md) 提供环境、数据与 checkpoint、逐级 A/B 命令、Nsight 用法和 5090 验收顺序。
- [Benchmark launcher](benchmarks/run_dl3dv_single_gpu.sh) 运行完整初始化前向、反向与优化器，输出 profile JSON。
- [Profiler 开关与三档流水](docs/PROFILING.md) 提供 PyTorch / Nsight 采集命令与已测 GPU 时间线。
- [RTX 4090 反向拆解与模块甘特图](docs/RTX4090-BACKWARD.md) 包含四次 Tensor Active 实测及同卡优化 A/B。
- [Benchmark 结果](benchmarks/results/) 包含选取的实测 JSON 和聚合画质记录。
- [上游说明](docs/UPSTREAM.md)、[数据格式](DATASETS.md)、[模型权重](MODEL_ZOO.md) 延续 ReSplat 的原始使用方式。

## 来源

基于 Haofei Xu、Daniel Barath、Andreas Geiger 和 Marc Pollefeys 的 [ReSplat](https://github.com/cvg/resplat) 公开提交 [`cae7ddc`](https://github.com/cvg/resplat/tree/cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1)。论文、引用与上游项目鸣谢见[上游说明](docs/UPSTREAM.md)，许可证见 [LICENSE](LICENSE) 与 [第三方声明](THIRD_PARTY_NOTICES.md)。
