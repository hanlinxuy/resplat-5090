# ReSplat profiler

在训练入口设置 `RESPLAT_PROFILE`。直接运行 `python -m src.main` 时默认关闭。Benchmark launcher 默认记录 timing，并把默认训练步数设为预热与采样之和。显式设置 `off` 可以关闭 profiler，`trainer.max_steps` 仍由训练参数控制。

| 开关 | 输出与用途 |
| --- | --- |
| `off` | 关闭采样和 trace |
| `timing` | `step-profile.json`。记录数据等待、CUDA 前向加 loss、反向、更新、显存峰值、模型形状和已启用的优化开关 |
| `torch` | 同一份 JSON，加 `torch-trace.json` 和按 shape 汇总的 `torch-trace.txt`。不依赖 Nsight |
| `nsys` | 启用同名 NVTX 阶段和 CUDA Profiler API 捕获窗口，用 `nsys profile` 保存可回放报告 |

`RESPLAT_PROFILE_WARMUP` 和 `RESPLAT_PROFILE_SAMPLES` 按 optimizer 更新计数，默认分别为 10 和 20。梯度累积组中的每个 microbatch 都单独记录。若训练提前正常结束，已经采到的结果会写盘并标记 `capture_complete: false`。建议只采 **2–4 次更新**以定位问题，再单独跑 timing 作为速度结果。CUDA event 会在采样 microbatch 末尾同步，Torch trace 另有 profiler 开销。

路径可用 `RESPLAT_PROFILE_JSON` 和 `RESPLAT_PROFILE_TORCH_TRACE` 指定。旧的 `RESPLAT_PROFILE_CAPTURE=1` 和 `RESPLAT_PROFILE_TORCH_TRACE` 在未显式设置模式时仍可启用对应后端。

## 采一份新 trace

先按 [Handoff](HANDOFF.md#1-环境与数据) 准备目录与模型。以下命令可追加相同的 Hydra 配置进行基线与候选对跑。

```bash
data=/path/to/dl3dv
checkpoint=/path/to/resplat-depth-base-352x640-60be7abf.pth
mkdir -p runs

RESPLAT_PROFILE=torch RESPLAT_PROFILE_WARMUP=10 RESPLAT_PROFILE_SAMPLES=3 \
  benchmarks/run_dl3dv_single_gpu.sh "$data" "$checkpoint" runs/torch 1

RESPLAT_PROFILE=nsys RESPLAT_PROFILE_WARMUP=10 RESPLAT_PROFILE_SAMPLES=3 \
  nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
    -o runs/nsys \
    benchmarks/run_dl3dv_single_gpu.sh "$data" "$checkpoint" runs/nsys-training 1

nsys export --type sqlite --output runs/nsys.sqlite runs/nsys.nsys-rep
python -m benchmarks.summarize_full_training_nsys runs/nsys.sqlite runs/nsys-summary.json \
  --chrome-trace runs/nsys-stages.json
```

Torch 和 Nsight 的标签都以 `resplat/` 开头。先看 `train_forward_loss`、`train_backward` 和 `train_update`，再展开 depth、cost volume、Point Transformer、KNN、各层局部 attention 和多视图卷积、Gaussian head、gsplat 与 VGG。Nsight 支持的设备可添加 `--gpu-metrics-devices=0` 采样 Tensor Active。逐核定位用 NCU。上传新跑数据时，阶段导出只保留 ReSplat 标签、相对 GPU 时间和 kernel 数。

## 三档已测流水

以下是 Thor 上 **256×448、B1、8+6 视角**的独立 NSYS 捕获，每档 2 个训练步。点击 JSON 下载后，在 [Perfetto](https://ui.perfetto.dev/) 里打开。时长是 CUDA launch 对应 GPU 核的时间区间，阶段间可能重叠，子区间包含在前向中。

| 阶段，中位数 ms | [local KNN 基线](../benchmarks/results/thor/pipelines/local.json) | [三项融合](../benchmarks/results/thor/pipelines/optimized.json) | [再开 BF16 感知和 fused AdamW](../benchmarks/results/thor/pipelines/mixed.json) |
| --- | ---: | ---: | ---: |
| 整步 GPU span | 2,602.75 | 2,438.36 | **1,417.82** |
| 前向 + loss | 1,409.51 | 1,341.89 | **521.61** |
| └ Point Transformer | 258.39 | **175.94** | 184.81 |
| └ VGG | 875.52 | 885.38 | **35.17** |
| └ gsplat | 7.36 | 7.21 | 7.48 |
| 反向 | 1,110.44 | 1,013.18 | **852.87** |
| 更新 | 79.89 | 79.92 | **39.85** |
| 每步 CUDA kernel 数 | 20,789 | 19,867 | **6,626** |
| Tensor Active | 4.22% | 4.47% | 8.21% |

三项融合主要缩短 Point Transformer 和反向。组合版进一步缩短 VGG 前向与更新后，反向占整步 GPU span 约 60%。这里的 B1 profiler 数字与 [有效 B32 的无 profiler 跑分](BENCHMARKS.md)使用不同口径。历史 trace 包含当时的阶段标签，新开关额外标出了 KNN、分块 cost volume 和逐层 attention / 卷积。
