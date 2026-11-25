# ReSplat 单卡训练交接

目标是在 RTX 5090 上复现完整六层 ReSplat Base 的初始化基线，逐项通过数值对照，再优化整步吞吐。初始化与递归 refinement 各自测量。此前 Thor、3090、4090 的结果用于校验实验口径，5090 尚无实测。

## 1. 环境与数据

先在 Linux 确认驱动和 CUDA Toolkit 可用。5090 使用 SM 12.0。下列 PyTorch 2.10.0、CUDA 12.8 和 gsplat 1.5.3 是已用于 3090/4090 的版本组合，也是首轮 5090 环境配方。[NVIDIA 架构表](https://developer.nvidia.com/cuda/gpus) · [PyTorch CUDA 版本](https://pytorch.org/get-started/previous-versions/)

```bash
nvidia-smi
nvcc --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

export TORCH_CUDA_ARCH_LIST=12.0
python -m pip install --no-build-isolation \
  'git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3'
python -m pip install --no-build-isolation ./src/model/encoder/pointops
python -c 'import torch, gsplat, pointops._C; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
```

从 [Model Zoo](../MODEL_ZOO.md) 下载官方 depth checkpoint `resplat-depth-base-352x640-60be7abf.pth`。历史 Thor 实验核对的 SHA-256 为 `60be7abfa9fcee53df5eee7b9a018cd44e595f2535ca571e373947c51c9e03bf`。真实 DL3DV 按 [DATASETS.md](../DATASETS.md) 转成训练 chunks。正式画质使用同一来源、同一哈希的 VGG 权重。短合成 benchmark 可以走内置固定种子分支。

## 2. 从原始速度逐级 A/B

以下在仓库根目录执行。每档使用 256×448、8 个输入与 6 个目标视角、128 个深度候选、六层 Point Transformer。保持 seed、样本顺序、视角、loss、checkpoint 和精度一致，重新启动进程，预热后记录 wall cadence、CUDA 前向/反向/更新和显存。多跑一次 local 参考测自然波动。

```bash
python -m benchmarks.make_synthetic_dl3dv synthetic-dl3dv
export CUDA_VISIBLE_DEVICES=0
export RESPLAT_SYNTHETIC_RANDOM_WEIGHTS=1
export RESPLAT_FP32_MATMUL_PRECISION=highest
export RESPLAT_STRICT_ACCUMULATION=1
export RESPLAT_PROFILE_WARMUP=10 RESPLAT_PROFILE_SAMPLES=20
export RESPLAT_PROFILE_PARAMETER_AUDIT=1

depth_checkpoint=pretrained/resplat-depth-base-352x640-60be7abf.pth

run_init() {
  local name=$1
  shift
  benchmarks/run_dl3dv_single_gpu.sh synthetic-dl3dv "$depth_checkpoint" \
    "runs/$name" 1 dataset.opencv_pose_format=false data_loader.train.num_workers=0 \
    data_loader.train.persistent_workers=false trainer.val_check_interval=1.0 "$@"
}

run_init upstream-global model.encoder.init_use_local_knn=false
run_init local
run_init local-repeat
run_init knn model.encoder.init_fused_knn_candidates=true

attention=(
  model.encoder.init_fused_knn_candidates=true
  model.encoder.init_fused_local_attention_rows=4
)
run_init knn-attention "${attention[@]}"
fused=("${attention[@]}" model.encoder.init_channels_last_conv=true)
run_init fused "${fused[@]}"
RESPLAT_PERCEPTUAL_CHANNELS_LAST=1 run_init vgg-layout "${fused[@]}"
RESPLAT_PERCEPTUAL_CHANNELS_LAST=1 RESPLAT_PERCEPTUAL_BF16=1 \
  run_init vgg-bf16 "${fused[@]}"
RESPLAT_PERCEPTUAL_CHANNELS_LAST=1 RESPLAT_PERCEPTUAL_BF16=1 \
  run_init mixed "${fused[@]}" optimizer.fused_adamw=true

python -m benchmarks.compare_training_profiles runs/local/step-profile.json runs/local-repeat/step-profile.json
python -m benchmarks.compare_training_profiles runs/local/step-profile.json runs/fused/step-profile.json
python -m benchmarks.compare_training_profiles runs/vgg-layout/step-profile.json runs/vgg-bf16/step-profile.json
python -m benchmarks.compare_training_profiles runs/vgg-bf16/step-profile.json runs/mixed/step-profile.json
```

这个矩阵用于新设备归因。[历史实测](BENCHMARKS.md)的部分阶段来自独立 A/B 窗口。上游全局 KNN 的数值单独记录，融合验收都对齐 local KNN 的邻域。短程对照关注逐步 loss 与抽样参数、梯度和 AdamW moments。脚本会拒绝样本对不齐或存在非有限值的结果。相对 L2 对接近零的尾数敏感，评估时一起看绝对误差和重复基线的波动。

## 3. 验收顺序

1. **索引与算子。** 对 B1/B2/B4 的 local KNN 比较全量索引，再测局部注意力输出、输入和参数梯度。收集耗时、显存、所选 Triton tile。800×800 配方先跑 D=19 的尾块数值测试。
2. **整步与并行。** 先跑 B1 匹配历史记录，再跑真 B2，最后在同一批 32 个样本上比较 B1×32 与 B2×16；各只更新一次，分别记录预热、每微步和整轮 wall。SM120 上逐扫 rows=1/2/4/8 和真 batch，决定是否保留重算。默认保持六层与 128 深度候选。
3. **带真实权重与数据的画质。** 从同一预训练权重启动等步数训练，固定留出场景和目标机位。用正式权重记录 loss 曲线与 PSNR、SSIM、LPIPS，抽查新视角渲染。BF16 感知分支在这一关通过后再定默认。当前短程证据还缺这项。
4. **把不同工作量单列。** 原生 800×800、4 层配方、每场景 80 万高斯和 refinement 都有独立计时。对于 gsplat，记实际可见 Gaussian-camera 对、tile 交叠数、分辨率和迭代数。对外给 scenes/s 时同时写出每场景 8 个输入和 6 个目标视角；10 万张累计输入图等于 12,500 个场景样本。

```bash
mkdir -p runs
git show cae7ddc:src/model/encoder/point_transformer/local_knn.py > runs/upstream-local-knn.py
python -m benchmarks.profile_dl3dv_local_knn --batch 2 \
  --original runs/upstream-local-knn.py --fused-knn runs/knn-b2.json
python -m benchmarks.profile_dl3dv_local_knn --transformer --batch 1 \
  --fused-local-rows 4 --channels-last-conv runs/point-b1.json
python -m benchmarks.check_fused_cost_volume
```

## 4. Profiling

正常跑分与 profiler 采样分开。使用 NVTX 对照 depth、Point Transformer、Gaussian head、渲染、反向和更新。Tensor Active 用 Nsight Systems 的设备采样，NCU 用来逐核诊断。整网 MFU 需按实跑 shape 核算训练 FLOPs，现有 Thor 的 MFU 仅覆盖 Point Transformer 子图。

```bash
RESPLAT_PROFILE_CAPTURE=1 RESPLAT_PROFILE_WARMUP=10 RESPLAT_PROFILE_SAMPLES=4 \
  nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
  -o runs/5090-local \
  benchmarks/run_dl3dv_single_gpu.sh synthetic-dl3dv "$depth_checkpoint" \
  runs/5090-local-profile 1 dataset.opencv_pose_format=false \
  data_loader.train.num_workers=0 data_loader.train.persistent_workers=false
```

需要 Tensor Active 时按设备支持添加 `--gpu-metrics-devices=0`。导出 SQLite 后可用 [整步分析脚本](../benchmarks/summarize_full_training_nsys.py) 和 [局部分析脚本](../benchmarks/summarize_local_knn_nsys.py)。

## 5. 448×224 的 80 万高斯分支

该实验对每个 112×56 latent 点输出 4×4 个子像素高斯，8 视角合计 `802816`。使用 [checkpoint 扩容脚本](../benchmarks/expand_gaussian_head_checkpoint.py) 把一个 Gaussian/latent 的初始化权重迁移到 16 个，并缩小初始尺度。这个实验冻结数据划分，训练完整初始化网络，使用项目中名为 mse 的 L1 图像损失。质量实验从官方 DL3DV 初始化权重继续拟合 Cafca v2 合成人脸。[Cafca 数据与许可](https://github.com/syntec-research/Cafca/blob/main/Dataset.md)

```bash
# Pointer to a locally prepared dataset with train/test chunks and view-index.json.
face_data=/path/to/cafca-448
source_ckpt=/path/to/trained-50k.ckpt
face_out=runs/face-803k
python -m benchmarks.expand_gaussian_head_checkpoint "$source_ckpt" runs/803k-head-warm-start.pth

env -u RESPLAT_SYNTHETIC_RANDOM_WEIGHTS -u RESPLAT_PERCEPTUAL_BF16 \
  RESPLAT_SYNTHETIC_QUALITY_FIXTURE="$face_data" RESPLAT_QUALITY_EVAL_STEPS=216,432,1080 \
  RESPLAT_PROFILE_JSON="$face_out/step-profile.json" RESPLAT_PROFILE_WARMUP=8 RESPLAT_PROFILE_SAMPLES=100 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m src.main +experiment=dl3dv \
  'loss=[mse]' '~loss.lpips' dataset/view_sampler=evaluation \
  "dataset.view_sampler.index_path=$face_data/view-index.json" \
  "dataset.roots=['$face_data']" dataset.view_sampler.num_context_views=8 \
  'dataset.image_shape=[224,448]' 'dataset.ori_image_shape=[224,448]' \
  dataset.near=0.25 dataset.far=2.0 dataset.augment=false dataset.opencv_pose_format=false \
  data_loader.train.batch_size=1 data_loader.train.num_workers=0 data_loader.train.persistent_workers=false \
  dataset.train_times_per_scene=6 model.encoder.shim_patch_size=4 \
  model.encoder.init_gaussian_multiple=16 model.encoder.init_use_checkpointing=true \
  model.encoder.init_fused_local_attention_rows=4 model.encoder.init_fused_knn_candidates=true \
  model.encoder.init_channels_last_conv=true optimizer.fused_adamw=true \
  optimizer.lr=0.00005 optimizer.warm_up_ratio=0.05 \
  checkpointing.pretrained_model=runs/803k-head-warm-start.pth checkpointing.no_strict_load=true \
  checkpointing.save_top_k=1 checkpointing.every_n_train_steps=216 \
  trainer.num_nodes=1 trainer.num_sanity_val_steps=0 trainer.val_check_interval=1.0 trainer.max_steps=216 \
  train.eval_model_every_n_val=0 wandb.mode=disabled "output_dir=$face_out"
```

80 万版的实际长训是从上述 216 步 checkpoint 再续 2,160 步。将 `checkpointing.pretrained_model` 指到该 checkpoint，并设 `trainer.max_steps=2160`。同一数据的 50K 对照保持 `init_gaussian_multiple=1`，直接载入训练好的 50K 起点。两项结果见[画质与吞吐记录](BENCHMARKS.md#其他独立负载)。手机端单帧延迟尚无实测。
